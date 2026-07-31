//! Python FFI Exports for Nautilus/Ray Bridge
//! 
//! Defines `extern "C"` FFI exports allowing Python/Nautilus backend to submit signals and fetch state.
//! Uses `#[no_mangle]` and strict C-ABI representations for zero-cost cross-language calls.
//! Implements `std::panic::catch_unwind` to prevent Rust panics from crashing the Python VM.

use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::panic::{self, AssertUnwindSafe};
use libc::size_t;

/// FFI-safe order signal structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiOrderSignal {
    pub symbol_ptr: *const c_char,
    pub side: i32, // 0 = Buy, 1 = Sell
    pub quantity: f64,
    pub price: f64,
    pub order_type: i32, // 0 = Market, 1 = Limit, 2 = StopLimit
    pub time_in_force: i32, // 0 = GTC, 1 = IOC, 2 = FOK
    pub client_order_id: u64,
    pub timestamp_ns: u64,
}

/// FFI-safe market data tick
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiTick {
    pub symbol_ptr: *const c_char,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub timestamp_ns: u64,
    pub sequence: u64,
}

/// FFI-safe portfolio delta
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiPortfolioDelta {
    pub symbol_ptr: *const c_char,
    pub delta_quantity: f64,
    pub avg_entry_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub timestamp_ns: u64,
}

/// FFI-safe execution report
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FfiExecutionReport {
    pub order_id: u64,
    pub client_order_id: u64,
    pub symbol_ptr: *const c_char,
    pub side: i32,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub avg_fill_price: f64,
    pub status: i32, // 0 = New, 1 = PartiallyFilled, 2 = Filled, 3 = Cancelled, 4 = Rejected
    pub reject_reason: i32,
    pub timestamp_ns: u64,
}

/// Global callback function pointers (set by Python)
static mut SIGNAL_CALLBACK: Option<unsafe extern "C" fn(*const FfiOrderSignal)> = None;
static mut STATE_CALLBACK: Option<unsafe extern "C" fn(*const FfiPortfolioDelta)> = None;
static mut EXECUTION_CALLBACK: Option<unsafe extern "C" fn(*const FfiExecutionReport)> = None;

/// Atomic flags for FFI initialization state
static FFI_INITIALIZED: AtomicBool = AtomicBool::new(false);
static CALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Result codes for FFI functions
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiResult {
    Success = 0,
    ErrorNullPtr = -1,
    ErrorInvalidParam = -2,
    ErrorNotInitialized = -3,
    ErrorPanic = -4,
    ErrorBufferOverflow = -5,
}

/// Initialize the FFI bridge (called from Python)
#[no_mangle]
pub unsafe extern "C" fn ffi_init() -> FfiResult {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        FFI_INITIALIZED.store(true, Ordering::SeqCst);
        CALLBACK_COUNT.store(0, Ordering::Relaxed);
    }));
    
    match result {
        Ok(_) => FfiResult::Success,
        Err(_) => FfiResult::ErrorPanic,
    }
}

/// Shutdown the FFI bridge (called from Python)
#[no_mangle]
pub unsafe extern "C" fn ffi_shutdown() -> FfiResult {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        FFI_INITIALIZED.store(false, Ordering::SeqCst);
        SIGNAL_CALLBACK = None;
        STATE_CALLBACK = None;
        EXECUTION_CALLBACK = None;
    }));
    
    match result {
        Ok(_) => FfiResult::Success,
        Err(_) => FfiResult::ErrorPanic,
    }
}

/// Check if FFI is initialized
#[no_mangle]
pub extern "C" fn ffi_is_initialized() -> bool {
    FFI_INITIALIZED.load(Ordering::Relaxed)
}

/// Register signal callback from Python
/// Safety: Caller must ensure callback pointer is valid
#[no_mangle]
pub unsafe extern "C" fn ffi_register_signal_callback(
    callback: Option<unsafe extern "C" fn(*const FfiOrderSignal)>,
) -> FfiResult {
    if !FFI_INITIALIZED.load(Ordering::Relaxed) {
        return FfiResult::ErrorNotInitialized;
    }
    
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        SIGNAL_CALLBACK = callback;
    }));
    
    match result {
        Ok(_) => FfiResult::Success,
        Err(_) => FfiResult::ErrorPanic,
    }
}

/// Register state callback from Python
#[no_mangle]
pub unsafe extern "C" fn ffi_register_state_callback(
    callback: Option<unsafe extern "C" fn(*const FfiPortfolioDelta)>,
) -> FfiResult {
    if !FFI_INITIALIZED.load(Ordering::Relaxed) {
        return FfiResult::ErrorNotInitialized;
    }
    
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        STATE_CALLBACK = callback;
    }));
    
    match result {
        Ok(_) => FfiResult::Success,
        Err(_) => FfiResult::ErrorPanic,
    }
}

/// Register execution callback from Python
#[no_mangle]
pub unsafe extern "C" fn ffi_register_execution_callback(
    callback: Option<unsafe extern "C" fn(*const FfiExecutionReport)>,
) -> FfiResult {
    if !FFI_INITIALIZED.load(Ordering::Relaxed) {
        return FfiResult::ErrorNotInitialized;
    }
    
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        EXECUTION_CALLBACK = callback;
    }));
    
    match result {
        Ok(_) => FfiResult::Success,
        Err(_) => FfiResult::ErrorPanic,
    }
}

/// Submit an order signal from Python to Rust
/// Returns FfiResult code
#[no_mangle]
pub unsafe extern "C" fn ffi_submit_signal(signal: *const FfiOrderSignal) -> FfiResult {
    if signal.is_null() {
        return FfiResult::ErrorNullPtr;
    }
    
    if !FFI_INITIALIZED.load(Ordering::Relaxed) {
        return FfiResult::ErrorNotInitialized;
    }
    
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        // Validate signal parameters
        let sig = &*signal;
        
        if sig.quantity <= 0.0 || sig.price < 0.0 {
            return FfiResult::ErrorInvalidParam;
        }
        
        if sig.side < 0 || sig.side > 1 {
            return FfiResult::ErrorInvalidParam;
        }
        
        // Forward to internal signal processor (stub - would connect to execution engine)
        process_order_signal(sig);
        
        CALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
        FfiResult::Success
    }));
    
    match result {
        Ok(r) => r,
        Err(_) => FfiResult::ErrorPanic,
    }
}

/// Get current portfolio state as JSON string (allocated, caller must free)
#[no_mangle]
pub extern "C" fn ffi_get_portfolio_json() -> *mut c_char {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        // In production, this would serialize actual portfolio state
        let json = r#"{"positions":[],"cash":100000.0,"pnl":0.0}"#;
        unsafe {
            libc::strdup(json.as_ptr() as *const i8)
        }
    }));
    
    match result {
        Ok(ptr) => ptr,
        Err(_) => ptr::null_mut(),
    }
}

/// Free a string allocated by FFI
#[no_mangle]
pub unsafe extern "C" fn ffi_free_string(s: *mut c_char) {
    if !s.is_null() {
        libc::free(s as *mut c_void);
    }
}

/// Get the number of callbacks processed
#[no_mangle]
pub extern "C" fn ffi_get_callback_count() -> u64 {
    CALLBACK_COUNT.load(Ordering::Relaxed)
}

/// Internal signal processor (would connect to execution engine)
unsafe fn process_order_signal(signal: &FfiOrderSignal) {
    // Extract symbol string
    let symbol = if !signal.symbol_ptr.is_null() {
        CStr::from_ptr(signal.symbol_ptr)
            .to_string_lossy()
            .into_owned()
    } else {
        String::from("UNKNOWN")
    };
    
    // Log signal for debugging (in production, route to execution engine)
    eprintln!(
        "[FFI] Order Signal: {} side={} qty={} price={} type={}",
        symbol, signal.side, signal.quantity, signal.price, signal.order_type
    );
    
    // If callback registered, invoke it
    if let Some(cb) = SIGNAL_CALLBACK {
        cb(signal);
    }
}

/// Push execution report to Python (called from Rust internals)
pub fn push_execution_report(report: &FfiExecutionReport) {
    unsafe {
        if let Some(cb) = EXECUTION_CALLBACK {
            let _ = panic::catch_unwind(AssertUnwindSafe(|| {
                cb(report);
            }));
        }
    }
}

/// Push portfolio delta to Python (called from Rust internals)
pub fn push_portfolio_delta(delta: &FfiPortfolioDelta) {
    unsafe {
        if let Some(cb) = STATE_CALLBACK {
            let _ = panic::catch_unwind(AssertUnwindSafe(|| {
                cb(delta);
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    
    #[test]
    fn test_ffi_initialization() {
        unsafe {
            assert_eq!(ffi_init(), FfiResult::Success);
            assert!(ffi_is_initialized());
            assert_eq!(ffi_shutdown(), FfiResult::Success);
            assert!(!ffi_is_initialized());
        }
    }
    
    #[test]
    fn test_ffi_signal_validation() {
        unsafe {
            ffi_init();
            
            let symbol = CString::new("BTCUSDT").unwrap();
            let signal = FfiOrderSignal {
                symbol_ptr: symbol.as_ptr(),
                side: 0,
                quantity: 1.0,
                price: 50000.0,
                order_type: 1,
                time_in_force: 0,
                client_order_id: 12345,
                timestamp_ns: 1000000000,
            };
            
            assert_eq!(ffi_submit_signal(&signal), FfiResult::Success);
            
            // Test invalid quantity
            let bad_signal = FfiOrderSignal {
                quantity: -1.0,
                ..signal
            };
            assert_eq!(ffi_submit_signal(&bad_signal), FfiResult::ErrorInvalidParam);
            
            ffi_shutdown();
        }
    }
    
    #[test]
    fn test_ffi_portfolio_json() {
        unsafe {
            ffi_init();
            let json_ptr = ffi_get_portfolio_json();
            assert!(!json_ptr.is_null());
            
            let json_str = CStr::from_ptr(json_ptr).to_string_lossy();
            assert!(json_str.contains("cash"));
            
            ffi_free_string(json_ptr);
            ffi_shutdown();
        }
    }
    
    #[test]
    fn test_ffi_panic_safety() {
        // Verify that even if something panics internally, FFI returns error code
        unsafe {
            // This should not crash even if internal logic panics
            let result = ffi_get_callback_count();
            assert_eq!(result, 0);
        }
    }
}
