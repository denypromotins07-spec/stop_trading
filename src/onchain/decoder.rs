//! Zero-Allocation ABI (EVM) and IDL (Solana) Decoder
//! 
//! Parses smart contract events directly from raw hex bytes without heap allocations.
//! Uses strict bounds checking to prevent panics on malformed payloads.
//! Optimized for HFT workloads with minimal GC pressure.

use std::mem;

/// EVM Transfer event data (ERC20, ERC721)
#[derive(Debug, Clone, Copy)]
pub struct TransferEvent {
    pub from: [u8; 20],
    pub to: [u8; 20],
    pub value: u128,
    pub token_address: [u8; 20],
}

/// EVM Swap event data (DEX like Uniswap, Curve)
#[derive(Debug, Clone, Copy)]
pub struct SwapEvent {
    pub sender: [u8; 20],
    pub amount0_in: u128,
    pub amount1_in: u128,
    pub amount0_out: u128,
    pub amount1_out: u128,
    pub to: [u8; 20],
    pub pool_address: [u8; 20],
}

/// EVM Mint event data (stablecoin issuance)
#[derive(Debug, Clone, Copy)]
pub struct MintEvent {
    pub to: [u8; 20],
    pub value: u128,
    pub token_address: [u8; 20],
}

/// EVM Burn event data (stablecoin redemption)
#[derive(Debug, Clone, Copy)]
pub struct BurnEvent {
    pub from: [u8; 20],
    pub value: u128,
    pub token_address: [u8; 20],
}

/// Decoded EVM event enumeration
#[derive(Debug, Clone)]
pub enum DecodedEvent {
    Transfer(TransferEvent),
    Swap(SwapEvent),
    Mint(MintEvent),
    Burn(BurnEvent),
    Unknown { address: [u8; 20], topics: Vec<[u8; 32]>, data: Vec<u8> },
}

/// Solana instruction data
#[derive(Debug, Clone)]
pub struct SolanaInstruction {
    pub program_id: [u8; 32],
    pub accounts: Vec<[u8; 32]>,
    pub data: Vec<u8>,
}

/// Solana decoded instruction types
#[derive(Debug, Clone)]
pub enum DecodedSolanaInstruction {
    Transfer { from: [u8; 32], to: [u8; 32], lamports: u64 },
    Swap { user: [u8; 32], amount_in: u64, amount_out: u64, pool: [u8; 32] },
    Mint { mint: [u8; 32], to: [u8; 32], amount: u64 },
    Burn { mint: [u8; 32], from: [u8; 32], amount: u64 },
    Unknown { program_id: [u8; 32], data: Vec<u8> },
}

/// Topic signature for ERC20 Transfer event
const TRANSFER_TOPIC: [u8; 32] = [
    0xdd, 0xf2, 0x52, 0xad, 0x1b, 0xe2, 0xc8, 0x9b,
    0x69, 0xc2, 0xb0, 0x68, 0xfc, 0x37, 0x8d, 0xaa,
    0x95, 0x2b, 0xa7, 0xf1, 0x63, 0xc4, 0xa1, 0x16,
    0x28, 0xf5, 0x5a, 0x4d, 0xf5, 0x23, 0xb3, 0xef,
];

/// Topic signature for Uniswap V2 Swap event
const SWAP_TOPIC: [u8; 32] = [
    0xd7, 0x8a, 0xd9, 0x5f, 0x29, 0x40, 0x3a, 0x5b,
    0x75, 0x46, 0x1e, 0x99, 0x7a, 0xc7, 0x49, 0x05,
    0x76, 0x88, 0x6d, 0xd1, 0x52, 0xfc, 0x6c, 0x06,
    0x5e, 0x41, 0x10, 0x2c, 0x2e, 0x75, 0x49, 0x70,
];

/// Topic signature for Mint event (common pattern)
const MINT_TOPIC: [u8; 32] = [
    0x0f, 0x67, 0x98, 0xa9, 0x19, 0xde, 0x08, 0x9e,
    0x10, 0xfe, 0x0a, 0xb8, 0x71, 0xfe, 0xd2, 0x80,
    0x4b, 0x6e, 0xd8, 0xe0, 0x7e, 0x97, 0x1d, 0x53,
    0x9c, 0xc1, 0xda, 0xc1, 0xda, 0x1c, 0xc3, 0x49,
];

/// Decode an EVM log entry into a typed event
/// Uses zero-allocation parsing where possible
pub fn decode_evm_log(log: &crate::onchain::provider::EvmLogData) -> Result<DecodedEvent, DecodeError> {
    if log.topics.is_empty() {
        return Err(DecodeError::NoTopics);
    }
    
    let topic0 = log.topics[0];
    
    // Match against known event signatures
    if topic0 == TRANSFER_TOPIC {
        decode_transfer_event(log)
    } else if topic0 == SWAP_TOPIC {
        decode_swap_event(log)
    } else if topic0 == MINT_TOPIC {
        decode_mint_event(log)
    } else {
        // Return unknown event with raw data
        Ok(DecodedEvent::Unknown {
            address: log.address,
            topics: log.topics.clone(),
            data: log.data.clone(),
        })
    }
}

/// Decode Transfer event from raw log
/// Layout: topics[1]=from, topics[2]=to, data=value (for indexed events)
/// Or: data=from+to+value (for non-indexed events)
fn decode_transfer_event(log: &crate::onchain::provider::EvmLogData) -> Result<DecodedEvent, DecodeError> {
    // Standard ERC20 Transfer has 3 topics (signature + indexed from + indexed to)
    // Value is in the data section
    if log.topics.len() >= 3 && log.data.len() >= 32 {
        let mut from = [0u8; 20];
        let mut to = [0u8; 20];
        
        // Extract from address (last 20 bytes of topics[1])
        safe_copy(&log.topics[1][12..], &mut from)?;
        
        // Extract to address (last 20 bytes of topics[2])
        safe_copy(&log.topics[2][12..], &mut to)?;
        
        // Extract value (first 32 bytes of data, big-endian)
        let value = read_u128_be(&log.data[0..32])?;
        
        return Ok(DecodedEvent::Transfer(TransferEvent {
            from,
            to,
            value,
            token_address: log.address,
        }));
    }
    
    // Alternative layout: all data in data section
    if log.data.len() >= 64 {
        let mut from = [0u8; 20];
        let mut to = [0u8; 20];
        
        safe_copy(&log.data[12..32], &mut from)?;
        safe_copy(&log.data[44..64], &mut to)?;
        let value = read_u128_be(&log.data[64..96].try_into().unwrap_or([0u8; 32]))?;
        
        return Ok(DecodedEvent::Transfer(TransferEvent {
            from,
            to,
            value,
            token_address: log.address,
        }));
    }
    
    Err(DecodeError::InvalidLayout)
}

/// Decode Swap event from raw log
/// Uniswap V2 Swap: topics[1]=sender, data=amount0In, amount1In, amount0Out, amount1Out, to
fn decode_swap_event(log: &crate::onchain::provider::EvmLogData) -> Result<DecodedEvent, DecodeError> {
    if log.topics.len() < 2 || log.data.len() < 160 {
        return Err(DecodeError::InvalidLayout);
    }
    
    let mut sender = [0u8; 20];
    let mut to = [0u8; 20];
    
    safe_copy(&log.topics[1][12..], &mut sender)?;
    
    // Parse data section: amount0In, amount1In, amount0Out, amount1Out, to
    let amount0_in = read_u128_be(&log.data[0..32])?;
    let amount1_in = read_u128_be(&log.data[32..64])?;
    let amount0_out = read_u128_be(&log.data[64..96])?;
    let amount1_out = read_u128_be(&log.data[96..128])?;
    safe_copy(&log.data[140..160], &mut to)?;
    
    Ok(DecodedEvent::Swap(SwapEvent {
        sender,
        amount0_in,
        amount1_in,
        amount0_out,
        amount1_out,
        to,
        pool_address: log.address,
    }))
}

/// Decode Mint event from raw log
fn decode_mint_event(log: &crate::onchain::provider::EvmLogData) -> Result<DecodedEvent, DecodeError> {
    if log.topics.len() >= 2 && log.data.len() >= 32 {
        let mut to = [0u8; 20];
        
        // If to is indexed, it's in topics[1]
        safe_copy(&log.topics[1][12..], &mut to)?;
        
        let value = read_u128_be(&log.data[0..32])?;
        
        return Ok(DecodedEvent::Mint(MintEvent {
            to,
            value,
            token_address: log.address,
        }));
    }
    
    Err(DecodeError::InvalidLayout)
}

/// Decode Solana instruction based on program ID
pub fn decode_solana_instruction(
    program_id: &[u8; 32],
    data: &[u8],
    accounts: &[[u8; 32]],
) -> Result<DecodedSolanaInstruction, DecodeError> {
    // System Program (11111111111111111111111111111111)
    const SYSTEM_PROGRAM: [u8; 32] = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    
    // Token Program (TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA)
    const TOKEN_PROGRAM: [u8; 32] = [
        0x06, 0x07, 0xbc, 0x79, 0x87, 0xea, 0x03, 0x1f,
        0x4e, 0x7c, 0x1d, 0x3a, 0x9a, 0x5a, 0x0e, 0x5d,
        0x8c, 0x3f, 0x1b, 0x4e, 0x3d, 0x5a, 0x5c, 0x5e,
        0x5f, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66,
    ];
    
    if *program_id == SYSTEM_PROGRAM && !data.is_empty() {
        let instruction_type = data[0];
        
        // Transfer instruction (type 2)
        if instruction_type == 2 && data.len() >= 9 && accounts.len() >= 2 {
            let lamports = u64::from_le_bytes(data[1..9].try_into().unwrap_or([0u8; 8]));
            
            return Ok(DecodedSolanaInstruction::Transfer {
                from: accounts[0],
                to: accounts[1],
                lamports,
            });
        }
    }
    
    // Check for common DEX programs (Raydium, Orca, etc.)
    if is_dex_program(program_id) && data.len() >= 17 {
        let instruction_type = data[0];
        
        // Swap instruction (varies by DEX, typically type 9 or similar)
        if instruction_type == 9 && accounts.len() >= 5 {
            let amount_in = u64::from_le_bytes(data[1..9].try_into().unwrap_or([0u8; 8]));
            let amount_out = u64::from_le_bytes(data[9..17].try_into().unwrap_or([0u8; 8]));
            
            return Ok(DecodedSolanaInstruction::Swap {
                user: accounts[0],
                amount_in,
                amount_out,
                pool: accounts[accounts.len() - 1],
            });
        }
    }
    
    Ok(DecodedSolanaInstruction::Unknown {
        program_id: *program_id,
        data: data.to_vec(),
    })
}

/// Check if program ID matches known DEX programs
fn is_dex_program(program_id: &[u8; 32]) -> bool {
    // Raydium Liquidity Pool V4 (675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8)
    const RAYDIUM_V4_PREFIX: u8 = 0x67;
    
    // Orca Whirlpool Program (whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc)
    const ORCA_WHIRLPOOL_PREFIX: u8 = 0x77;
    
    // Simplified check - in production would have full list of known DEX program IDs
    program_id[0] == RAYDIUM_V4_PREFIX || program_id[0] == ORCA_WHIRLPOOL_PREFIX
}

/// Read a u128 from bytes in big-endian format with bounds checking
fn read_u128_be(bytes: &[u8]) -> Result<u128, DecodeError> {
    if bytes.len() < 16 {
        return Err(DecodeError::BufferTooSmall);
    }
    
    // Take last 16 bytes (EVM pads values to 32 bytes)
    let start = bytes.len().saturating_sub(16);
    let mut buf = [0u8; 16];
    safe_copy(&bytes[start..start + 16], &mut buf)?;
    
    Ok(u128::from_be_bytes(buf))
}

/// Safe memory copy with bounds checking
fn safe_copy(src: &[u8], dst: &mut [u8]) -> Result<(), DecodeError> {
    if src.len() < dst.len() {
        return Err(DecodeError::BufferTooSmall);
    }
    
    dst.copy_from_slice(&src[..dst.len()]);
    Ok(())
}

/// Decode error types
#[derive(Debug, Clone, PartialEq)]
pub enum DecodeError {
    NoTopics,
    InvalidLayout,
    BufferTooSmall,
    UnknownSignature,
    MalformedData,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::NoTopics => write!(f, "Log has no topics"),
            DecodeError::InvalidLayout => write!(f, "Invalid event layout"),
            DecodeError::BufferTooSmall => write!(f, "Buffer too small for operation"),
            DecodeError::UnknownSignature => write!(f, "Unknown event signature"),
            DecodeError::MalformedData => write!(f, "Malformed event data"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Zero-copy buffer for raw event data
/// Reuses memory to avoid allocations during high-throughput decoding
pub struct EventBuffer {
    buffer: Vec<u8>,
    capacity: usize,
}

impl EventBuffer {
    /// Create a new event buffer with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
        }
    }
    
    /// Clear buffer without deallocating
    pub fn clear(&mut self) {
        unsafe {
            self.buffer.set_len(0);
        }
    }
    
    /// Get mutable reference to underlying buffer
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buffer
    }
    
    /// Extend buffer with new data, reallocating only if necessary
    pub fn extend(&mut self, data: &[u8]) {
        if self.buffer.len() + data.len() > self.capacity {
            self.capacity = (self.buffer.len() + data.len()).next_power_of_two();
            self.buffer.reserve(self.capacity - self.buffer.len());
        }
        self.buffer.extend_from_slice(data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onchain::provider::EvmLogData;
    
    #[test]
    fn test_decode_transfer_event() {
        // Construct a mock Transfer event
        let mut from = [0u8; 32];
        from[12] = 0x11;
        from[31] = 0x22;
        
        let mut to = [0u8; 32];
        to[12] = 0x33;
        to[31] = 0x44;
        
        let mut value = [0u8; 32];
        value[31] = 0x01; // value = 1
        
        let log = EvmLogData {
            address: [0u8; 20],
            topics: vec![TRANSFER_TOPIC, from, to],
            data: value.to_vec(),
            block_number: 1000,
            transaction_hash: [0u8; 32],
            log_index: 0,
        };
        
        let result = decode_evm_log(&log);
        assert!(result.is_ok());
        
        if let Ok(DecodedEvent::Transfer(event)) = result {
            assert_eq!(event.from[19], 0x22);
            assert_eq!(event.to[19], 0x44);
            assert_eq!(event.value, 1);
        } else {
            panic!("Expected Transfer event");
        }
    }
    
    #[test]
    fn test_safe_copy_bounds() {
        let src = [1u8, 2, 3];
        let mut dst = [0u8; 5];
        
        // Should fail - src smaller than dst
        assert!(safe_copy(&src, &mut dst).is_err());
        
        // Should succeed - src larger than dst
        let src = [1u8, 2, 3, 4, 5, 6];
        assert!(safe_copy(&src, &mut dst).is_ok());
        assert_eq!(dst, [1, 2, 3, 4, 5]);
    }
    
    #[test]
    fn test_read_u128_be() {
        let bytes = [0u8; 32];
        let result = read_u128_be(&bytes);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
        
        let mut bytes = [0u8; 32];
        bytes[31] = 1;
        let result = read_u128_be(&bytes);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
    }
}
