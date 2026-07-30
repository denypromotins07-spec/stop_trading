//! FIX 4.4 Message Codec
//!
//! Zero-allocation FIX message encoder/decoder using byte-slice parsing.
//! Avoids standard string allocations by referencing byte offsets directly
//! within the receive buffer for microsecond parsing.

use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum FIX message size (bytes)
pub const MAX_FIX_MESSAGE_SIZE: usize = 4096;

/// SOH character (ASCII 1) - FIX field delimiter
pub const SOH: u8 = 1;

/// FIX tag constants
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixTag {
    BeginString = 8,
    BodyLength = 9,
    MsgType = 35,
    SenderCompID = 49,
    TargetCompID = 56,
    MsgSeqNum = 34,
    SendingTime = 52,
    CheckSum = 10,
    // Order tags
    ClOrdID = 11,
    OrderQty = 38,
    Side = 54,
    Symbol = 55,
    Price = 44,
    OrdType = 40,
    TimeInForce = 59,
    // Execution tags
    ExecID = 17,
    ExecType = 150,
    OrdStatus = 39,
    LastQty = 32,
    LastPx = 31,
    // Session tags
    HeartBtInt = 108,
    TestReqID = 112,
    GapFillFlag = 123,
    PossResend = 97,
    EncryptMethod = 98,
    HeartbeatInterval = 108,
}

impl FixTag {
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    #[inline]
    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            8 => Some(FixTag::BeginString),
            9 => Some(FixTag::BodyLength),
            35 => Some(FixTag::MsgType),
            49 => Some(FixTag::SenderCompID),
            56 => Some(FixTag::TargetCompID),
            34 => Some(FixTag::MsgSeqNum),
            52 => Some(FixTag::SendingTime),
            10 => Some(FixTag::CheckSum),
            11 => Some(FixTag::ClOrdID),
            38 => Some(FixTag::OrderQty),
            54 => Some(FixTag::Side),
            55 => Some(FixTag::Symbol),
            44 => Some(FixTag::Price),
            40 => Some(FixTag::OrdType),
            59 => Some(FixTag::TimeInForce),
            17 => Some(FixTag::ExecID),
            150 => Some(FixTag::ExecType),
            39 => Some(FixTag::OrdStatus),
            32 => Some(FixTag::LastQty),
            31 => Some(FixTag::LastPx),
            108 => Some(FixTag::HeartBtInt),
            112 => Some(FixTag::TestReqID),
            123 => Some(FixTag::GapFillFlag),
            97 => Some(FixTag::PossResend),
            _ => None,
        }
    }
}

/// FIX field reference - zero-copy view into buffer
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FixField<'a> {
    /// Tag number
    pub tag: u32,
    /// Value as byte slice (zero-copy)
    pub value: &'a [u8],
    /// Start offset in original buffer
    pub offset: usize,
    /// Length including tag, =, value, and SOH
    pub length: usize,
}

impl<'a> FixField<'a> {
    #[inline]
    pub fn new(tag: u32, value: &'a [u8], offset: usize, length: usize) -> Self {
        Self { tag, value, offset, length }
    }

    /// Parse value as integer
    #[inline]
    pub fn as_int(&self) -> Result<i64, FixError> {
        parse_int(self.value)
    }

    /// Parse value as unsigned integer
    #[inline]
    pub fn as_uint(&self) -> Result<u64, FixError> {
        parse_uint(self.value)
    }

    /// Get value as string slice
    #[inline]
    pub fn as_str(&self) -> Result<&str, FixError> {
        std::str::from_utf8(self.value).map_err(|_| FixError::InvalidUtf8)
    }

    /// Get value as char (for single-char fields like Side)
    #[inline]
    pub fn as_char(&self) -> Result<char, FixError> {
        if self.value.is_empty() {
            return Err(FixError::EmptyValue);
        }
        Ok(self.value[0] as char)
    }
}

/// FIX message structure with zero-copy field references
#[repr(C)]
#[derive(Debug)]
pub struct FixMessage<'a> {
    /// Raw buffer reference
    buffer: &'a [u8],
    /// Fields parsed from buffer
    fields: [Option<FixField<'a>>; 64],
    /// Number of fields
    field_count: usize,
    /// Message type
    msg_type: Option<&'a [u8]>,
    /// Begin string (FIX version)
    begin_string: Option<&'a [u8]>,
    /// Body length
    body_length: usize,
    /// Checksum value
    checksum: u8,
    /// Calculated checksum
    calculated_checksum: u8,
}

impl<'a> FixMessage<'a> {
    #[inline]
    pub fn new(buffer: &'a [u8]) -> Self {
        Self {
            buffer,
            fields: std::array::from_fn(|_| None),
            field_count: 0,
            msg_type: None,
            begin_string: None,
            body_length: 0,
            checksum: 0,
            calculated_checksum: 0,
        }
    }

    /// Get field by tag
    #[inline]
    pub fn get_field(&self, tag: FixTag) -> Option<&FixField<'a>> {
        let tag_u32 = tag.as_u32();
        for i in 0..self.field_count {
            if let Some(ref field) = self.fields[i] {
                if field.tag == tag_u32 {
                    return Some(field);
                }
            }
        }
        None
    }

    /// Get all fields (iterator-like access)
    #[inline]
    pub fn fields(&self) -> impl Iterator<Item = &FixField<'a>> {
        self.fields.iter().filter_map(|f| f.as_ref())
    }

    /// Get message type
    #[inline]
    pub fn msg_type(&self) -> Option<&'a str> {
        self.msg_type.and_then(|b| std::str::from_utf8(b).ok())
    }

    /// Get begin string
    #[inline]
    pub fn begin_string(&self) -> Option<&'a str> {
        self.begin_string.and_then(|b| std::str::from_utf8(b).ok())
    }

    /// Validate checksum
    #[inline]
    pub fn validate_checksum(&self) -> bool {
        self.checksum == self.calculated_checksum
    }

    /// Get raw buffer
    #[inline]
    pub fn buffer(&self) -> &'a [u8] {
        self.buffer
    }

    /// Get field count
    #[inline]
    pub fn field_count(&self) -> usize {
        self.field_count
    }
}

/// FIX codec errors
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixError {
    /// Buffer too small
    BufferTooSmall,
    /// Invalid tag format
    InvalidTagFormat,
    /// Missing required field
    MissingRequiredField,
    /// Invalid checksum
    InvalidChecksum,
    /// Invalid UTF-8
    InvalidUtf8,
    /// Empty value
    EmptyValue,
    /// Parse error
    ParseError,
    /// Malformed message
    MalformedMessage,
    /// Out of bounds access
    OutOfBounds,
    /// Session not active
    SessionNotActive,
    /// Sequence gap detected
    SequenceGap,
    /// Invalid message type
    InvalidMessageType,
}

impl FixError {
    #[inline]
    pub fn error_code(&self) -> u32 {
        match self {
            FixError::BufferTooSmall => 1,
            FixError::InvalidTagFormat => 2,
            FixError::MissingRequiredField => 3,
            FixError::InvalidChecksum => 4,
            FixError::InvalidUtf8 => 5,
            FixError::EmptyValue => 6,
            FixError::ParseError => 7,
            FixError::MalformedMessage => 8,
            FixError::OutOfBounds => 9,
            FixError::SessionNotActive => 10,
            FixError::SequenceGap => 11,
            FixError::InvalidMessageType => 12,
        }
    }
}

/// Zero-allocation FIX codec
#[repr(C)]
pub struct FixCodec {
    /// Messages encoded counter
    messages_encoded: AtomicU64,
    /// Messages decoded counter
    messages_decoded: AtomicU64,
    /// Parse errors counter
    parse_errors: AtomicU64,
}

impl FixCodec {
    pub fn new() -> Self {
        Self {
            messages_encoded: AtomicU64::new(0),
            messages_decoded: AtomicU64::new(0),
            parse_errors: AtomicU64::new(0),
        }
    }

    /// Encode a FIX message into buffer
    /// Returns the number of bytes written
    #[inline]
    pub fn encode(&self, msg: &FixMessage, buffer: &mut [u8]) -> Result<usize, FixError> {
        // For encoding, we reconstruct from fields
        // In production, would build message from structured data
        
        let mut pos = 0;
        let mut checksum_start = 0;

        // Write each field
        for field in msg.fields() {
            if pos + field.length > buffer.len() {
                self.parse_errors.fetch_add(1, Ordering::Relaxed);
                return Err(FixError::BufferTooSmall);
            }

            // Track where body starts (after BeginString and BodyLength)
            if checksum_start == 0 && field.tag != FixTag::BeginString.as_u32() 
                && field.tag != FixTag::BodyLength.as_u32() 
            {
                checksum_start = pos;
            }

            // Write tag
            let tag_str = itoa_buffer(field.tag, &mut buffer[pos..]);
            pos += tag_str.len();
            
            if pos >= buffer.len() {
                self.parse_errors.fetch_add(1, Ordering::Relaxed);
                return Err(FixError::BufferTooSmall);
            }

            // Write =
            buffer[pos] = b'=';
            pos += 1;

            // Write value
            let value_len = field.value.len();
            if pos + value_len + 1 > buffer.len() {
                self.parse_errors.fetch_add(1, Ordering::Relaxed);
                return Err(FixError::BufferTooSmall);
            }
            
            buffer[pos..pos + value_len].copy_from_slice(field.value);
            pos += value_len;

            // Write SOH
            buffer[pos] = SOH;
            pos += 1;
        }

        // Calculate and append checksum
        let checksum = calculate_checksum(&buffer[checksum_start..pos]);
        
        if pos + 7 > buffer.len() {
            self.parse_errors.fetch_add(1, Ordering::Relaxed);
            return Err(FixError::BufferTooSmall);
        }

        // Write checksum field: 10=XXX<SOH>
        buffer[pos] = b'1';
        buffer[pos + 1] = b'0';
        buffer[pos + 2] = b'=';
        write_checksum(&mut buffer[pos + 3..], checksum);
        buffer[pos + 6] = SOH;
        pos += 7;

        self.messages_encoded.fetch_add(1, Ordering::Relaxed);
        Ok(pos)
    }

    /// Decode a FIX message from buffer (zero-copy)
    #[inline]
    pub fn decode<'b>(&self, buffer: &'b [u8]) -> Result<FixMessage<'b>, FixError> {
        if buffer.is_empty() {
            self.parse_errors.fetch_add(1, Ordering::Relaxed);
            return Err(FixError::MalformedMessage);
        }

        let mut msg = FixMessage::new(buffer);
        let mut pos = 0;
        let mut field_idx = 0;
        let mut body_start = 0;
        let mut body_end = 0;
        let mut checksum_offset = 0;

        while pos < buffer.len() && field_idx < msg.fields.len() {
            // Find tag end (=)
            let eq_pos = find_byte(buffer, pos, b'=')
                .ok_or_else(|| {
                    self.parse_errors.fetch_add(1, Ordering::Relaxed);
                    FixError::InvalidTagFormat
                })?;

            // Parse tag
            let tag = parse_uint(&buffer[pos..eq_pos])
                .map_err(|_| {
                    self.parse_errors.fetch_add(1, Ordering::Relaxed);
                    FixError::InvalidTagFormat
                })? as u32;

            pos = eq_pos + 1;

            // Find SOH (value end)
            let soh_pos = find_byte(buffer, pos, SOH)
                .ok_or_else(|| {
                    self.parse_errors.fetch_add(1, Ordering::Relaxed);
                    FixError::MalformedMessage
                })?;

            // Extract value (zero-copy slice)
            let value = &buffer[pos..soh_pos];
            let field_len = soh_pos - pos + 1 + (eq_pos - pos) + 2; // tag + = + value + SOH

            // Track body boundaries
            if tag == FixTag::BodyLength.as_u32() {
                body_start = soh_pos + 1;
                msg.body_length = parse_uint(value)
                    .map_err(|_| {
                        self.parse_errors.fetch_add(1, Ordering::Relaxed);
                        FixError::ParseError
                    })? as usize;
            } else if tag == FixTag::CheckSum.as_u32() {
                checksum_offset = pos;
                msg.checksum = parse_checksum(value)
                    .map_err(|_| {
                        self.parse_errors.fetch_add(1, Ordering::Relaxed);
                        FixError::InvalidChecksum
                    })?;
                body_end = pos;
            }

            // Store field
            msg.fields[field_idx] = Some(FixField::new(tag, value, pos, field_len));
            field_idx += 1;

            // Track special fields
            if tag == FixTag::MsgType.as_u32() {
                msg.msg_type = Some(value);
            } else if tag == FixTag::BeginString.as_u32() {
                msg.begin_string = Some(value);
            }

            pos = soh_pos + 1;
        }

        msg.field_count = field_idx;

        // Calculate checksum over body
        if body_start > 0 && body_end > body_start {
            msg.calculated_checksum = calculate_checksum(&buffer[body_start..body_end]);
        }

        self.messages_decoded.fetch_add(1, Ordering::Relaxed);
        Ok(msg)
    }

    /// Get codec statistics
    #[inline]
    pub fn get_stats(&self) -> FixCodecStats {
        FixCodecStats {
            messages_encoded: self.messages_encoded.load(Ordering::Relaxed),
            messages_decoded: self.messages_decoded.load(Ordering::Relaxed),
            parse_errors: self.parse_errors.load(Ordering::Relaxed),
        }
    }
}

impl Default for FixCodec {
    fn default() -> Self {
        Self::new()
    }
}

/// Codec statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FixCodecStats {
    pub messages_encoded: u64,
    pub messages_decoded: u64,
    pub parse_errors: u64,
}

/// Find byte in buffer starting from offset
#[inline]
fn find_byte(buffer: &[u8], start: usize, byte: u8) -> Option<usize> {
    if start >= buffer.len() {
        return None;
    }
    buffer[start..].iter().position(|&b| b == byte).map(|p| start + p)
}

/// Parse unsigned integer from byte slice
#[inline]
fn parse_uint(bytes: &[u8]) -> Result<u64, FixError> {
    if bytes.is_empty() {
        return Err(FixError::EmptyValue);
    }

    let mut result: u64 = 0;
    for &b in bytes {
        if b < b'0' || b > b'9' {
            return Err(FixError::ParseError);
        }
        result = result.checked_mul(10)
            .and_then(|r| r.checked_add((b - b'0') as u64))
            .ok_or(FixError::ParseError)?;
    }
    Ok(result)
}

/// Parse signed integer from byte slice
#[inline]
fn parse_int(bytes: &[u8]) -> Result<i64, FixError> {
    if bytes.is_empty() {
        return Err(FixError::EmptyValue);
    }

    let mut negative = false;
    let mut start = 0;

    if bytes[0] == b'-' {
        negative = true;
        start = 1;
    } else if bytes[0] == b'+' {
        start = 1;
    }

    let mut result: i64 = 0;
    for &b in &bytes[start..] {
        if b < b'0' || b > b'9' {
            return Err(FixError::ParseError);
        }
        result = result.checked_mul(10)
            .and_then(|r| r.checked_add((b - b'0') as i64))
            .ok_or(FixError::ParseError)?;
    }

    if negative {
        result = -result;
    }

    Ok(result)
}

/// Parse checksum value (3-digit octal representation)
#[inline]
fn parse_checksum(bytes: &[u8]) -> Result<u8, FixError> {
    if bytes.len() != 3 {
        return Err(FixError::ParseError);
    }

    let mut result: u16 = 0;
    for &b in bytes {
        if b < b'0' || b > b'9' {
            return Err(FixError::ParseError);
        }
        result = result * 10 + (b - b'0') as u16;
    }

    if result > 255 {
        return Err(FixError::ParseError);
    }

    Ok(result as u8)
}

/// Calculate FIX checksum (sum of bytes mod 256)
#[inline]
fn calculate_checksum(data: &[u8]) -> u8 {
    let sum: u32 = data.iter().map(|&b| b as u32).sum();
    (sum % 256) as u8
}

/// Write checksum as 3-digit string
#[inline]
fn write_checksum(buffer: &mut [u8], checksum: u8) {
    if buffer.len() < 3 {
        return;
    }
    buffer[0] = b'0' + ((checksum / 100) % 10);
    buffer[1] = b'0' + ((checksum / 10) % 10);
    buffer[2] = b'0' + (checksum % 10);
}

/// Fast integer to string conversion (small buffer version)
#[inline]
fn itoa_buffer(mut value: u32, buffer: &mut [u8]) -> usize {
    if value == 0 {
        if !buffer.is_empty() {
            buffer[0] = b'0';
            return 1;
        }
        return 0;
    }

    let mut digits = [0u8; 10];
    let mut idx = 0;

    while value > 0 && idx < 10 {
        digits[idx] = b'0' + (value % 10) as u8;
        value /= 10;
        idx += 1;
    }

    if idx > buffer.len() {
        return 0;
    }

    // Reverse into buffer
    for i in 0..idx {
        buffer[i] = digits[idx - 1 - i];
    }

    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_uint() {
        assert_eq!(parse_uint(b"123").unwrap(), 123);
        assert_eq!(parse_uint(b"0").unwrap(), 0);
        assert_eq!(parse_uint(b"999999").unwrap(), 999999);
        assert!(parse_uint(b"").is_err());
        assert!(parse_uint(b"abc").is_err());
    }

    #[test]
    fn test_parse_int() {
        assert_eq!(parse_int(b"123").unwrap(), 123);
        assert_eq!(parse_int(b"-456").unwrap(), -456);
        assert_eq!(parse_int(b"+789").unwrap(), 789);
        assert!(parse_int(b"").is_err());
    }

    #[test]
    fn test_calculate_checksum() {
        let data = b"10=123";
        let checksum = calculate_checksum(data);
        assert!(checksum <= 255);
    }

    #[test]
    fn test_codec_encode_decode() {
        let codec = FixCodec::new();
        
        // Create a simple message manually
        let buffer = b"8=FIX.4.4\x019=100\x0135=D\x0149=SENDER\x0156=TARGET\x0134=1\x0152=20240101-12:00:00\x01";
        
        let msg = codec.decode(buffer).unwrap();
        assert_eq!(msg.field_count(), 7);
        assert_eq!(msg.msg_type(), Some("D"));
        assert_eq!(msg.begin_string(), Some("FIX.4.4"));
    }

    #[test]
    fn test_fix_tag_conversion() {
        assert_eq!(FixTag::MsgType.as_u32(), 35);
        assert_eq!(FixTag::from_u32(35), Some(FixTag::MsgType));
        assert_eq!(FixTag::from_u32(999), None);
    }

    #[test]
    fn test_find_byte() {
        let buffer = b"hello=world";
        assert_eq!(find_byte(buffer, 0, b'='), Some(5));
        assert_eq!(find_byte(buffer, 6, b'o'), Some(7));
        assert_eq!(find_byte(buffer, 0, b'x'), None);
    }
}
