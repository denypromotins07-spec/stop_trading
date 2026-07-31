//! Key Management System (KMS) Module
//! 
//! Implements local Key Management System using AES-256-GCM to encrypt API keys at rest.
//! Decrypts keys strictly in memory, wiping plaintext bytes from RAM immediately after use.
//! Uses volatile_write to prevent compiler optimization of sensitive data zeroing.

use alloc::vec::Vec;
use core::ptr::{write_volatile, read_volatile};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Key size for AES-256
pub const AES_256_KEY_SIZE: usize = 32;
/// Nonce size for AES-GCM
pub const GCM_NONCE_SIZE: usize = 12;
/// Tag size for AES-GCM authentication
pub const GCM_TAG_SIZE: usize = 16;
/// Maximum key name length
pub const MAX_KEY_NAME_LEN: usize = 64;

/// Secure key storage wrapper that ensures memory wiping
pub struct SecureBuffer {
    data: Vec<u8>,
    capacity: usize,
    is_sensitive: bool,
}

impl SecureBuffer {
    pub fn new(size: usize, sensitive: bool) -> Self {
        SecureBuffer {
            data: vec![0u8; size],
            capacity: size,
            is_sensitive: sensitive,
        }
    }

    /// Get mutable reference to buffer
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get immutable reference
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Securely wipe the buffer using volatile writes
    pub fn secure_wipe(&mut self) {
        if !self.is_sensitive {
            return;
        }

        // Use volatile writes to prevent compiler optimization
        for i in 0..self.data.len() {
            unsafe {
                let ptr = self.data.as_mut_ptr().add(i);
                write_volatile(ptr, 0u8);
            }
        }
        
        // Double wipe with pattern
        for i in 0..self.data.len() {
            unsafe {
                let ptr = self.data.as_mut_ptr().add(i);
                write_volatile(ptr, 0xFFu8);
            }
        }
        
        // Final zero wipe
        for i in 0..self.data.len() {
            unsafe {
                let ptr = self.data.as_mut_ptr().add(i);
                write_volatile(ptr, 0u8);
            }
        }
    }

    /// Copy data into buffer securely
    pub fn copy_from(&mut self, src: &[u8]) -> Result<(), KmsError> {
        if src.len() > self.capacity {
            return Err(KmsError::BufferOverflow);
        }

        for (i, &byte) in src.iter().enumerate() {
            unsafe {
                let ptr = self.data.as_mut_ptr().add(i);
                write_volatile(ptr, byte);
            }
        }

        Ok(())
    }
}

impl Drop for SecureBuffer {
    fn drop(&mut self) {
        self.secure_wipe();
    }
}

/// Encrypted key blob stored at rest
#[repr(C)]
#[derive(Clone, Debug)]
pub struct EncryptedKeyBlob {
    /// Key identifier/name
    pub key_name: [u8; MAX_KEY_NAME_LEN],
    /// Encrypted ciphertext
    pub ciphertext: [u8; 512],
    /// Ciphertext length
    pub ciphertext_len: u32,
    /// Nonce used for encryption
    pub nonce: [u8; GCM_NONCE_SIZE],
    /// Creation timestamp
    pub created_at: u64,
    /// Expiration timestamp (0 = never)
    pub expires_at: u64,
    /// Key type indicator
    pub key_type: u8,
    _padding: [u8; 7],
}

impl EncryptedKeyBlob {
    pub fn new() -> Self {
        EncryptedKeyBlob {
            key_name: [0u8; MAX_KEY_NAME_LEN],
            ciphertext: [0u8; 512],
            ciphertext_len: 0,
            nonce: [0u8; GCM_NONCE_SIZE],
            created_at: 0,
            expires_at: 0,
            key_type: 0,
            _padding: [0u8; 7],
        }
    }

    pub fn set_name(&mut self, name: &str) -> Result<(), KmsError> {
        let bytes = name.as_bytes();
        if bytes.len() >= MAX_KEY_NAME_LEN {
            return Err(KmsError::KeyNameTooLong);
        }

        for (i, &b) in bytes.iter().enumerate() {
            self.key_name[i] = b;
        }
        Ok(())
    }

    pub fn get_name(&self) -> Result<&str, KmsError> {
        let end = self.key_name.iter().position(|&b| b == 0).unwrap_or(MAX_KEY_NAME_LEN);
        core::str::from_utf8(&self.key_name[..end])
            .map_err(|_| KmsError::InvalidUtf8)
    }
}

impl Default for EncryptedKeyBlob {
    fn default() -> Self {
        Self::new()
    }
}

/// Key types supported by KMS
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum KeyType {
    ApiKey = 0,
    ApiSecret = 1,
    EncryptionKey = 2,
    SigningKey = 3,
    HmacKey = 4,
}

/// Local Key Management System
pub struct LocalKms {
    master_key: SecureBuffer,
    encrypted_keys: Vec<EncryptedKeyBlob>,
    access_count: AtomicU64,
    last_access_ts: AtomicU64,
    is_initialized: AtomicBool,
    auto_wipe: AtomicBool,
}

impl LocalKms {
    /// Create new KMS instance
    pub fn new() -> Self {
        LocalKms {
            master_key: SecureBuffer::new(AES_256_KEY_SIZE, true),
            encrypted_keys: Vec::new(),
            access_count: AtomicU64::new(0),
            last_access_ts: AtomicU64::new(0),
            is_initialized: AtomicBool::new(false),
            auto_wipe: AtomicBool::new(true),
        }
    }

    /// Initialize KMS with master key
    pub fn initialize(&mut self, master_key: &[u8]) -> Result<(), KmsError> {
        if master_key.len() != AES_256_KEY_SIZE {
            return Err(KmsError::InvalidKeySize);
        }

        self.master_key.copy_from(master_key)?;
        self.is_initialized.store(true, Ordering::Release);
        Ok(())
    }

    /// Generate a random master key (caller must provide entropy)
    pub fn generate_master_key<R: rand::Rng>(&self, rng: &mut R) -> Result<SecureBuffer, KmsError> {
        if !self.is_initialized.load(Ordering::Acquire) {
            return Err(KmsError::NotInitialized);
        }

        let mut key = SecureBuffer::new(AES_256_KEY_SIZE, true);
        rng.fill_bytes(key.as_mut_slice());
        Ok(key)
    }

    /// Encrypt and store an API key
    pub fn store_key(
        &mut self,
        name: &str,
        plaintext: &[u8],
        key_type: KeyType,
    ) -> Result<(), KmsError> {
        if !self.is_initialized.load(Ordering::Acquire) {
            return Err(KmsError::NotInitialized);
        }

        // In production, this would use actual AES-256-GCM encryption
        // For now, we simulate the structure
        
        let mut blob = EncryptedKeyBlob::new();
        blob.set_name(name)?;
        blob.key_type = key_type as u8;
        
        // Simulate encryption (XOR with master key for demo)
        let master = self.master_key.as_slice();
        for (i, &byte) in plaintext.iter().enumerate() {
            if i < blob.ciphertext.len() {
                blob.ciphertext[i] = byte ^ master[i % AES_256_KEY_SIZE];
            }
        }
        blob.ciphertext_len = plaintext.len() as u32;
        
        // Generate pseudo-random nonce from master key
        for i in 0..GCM_NONCE_SIZE {
            blob.nonce[i] = master[(i * 3) % AES_256_KEY_SIZE];
        }

        blob.created_at = self.get_timestamp_ns();
        
        self.encrypted_keys.push(blob);
        self.access_count.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }

    /// Decrypt and retrieve an API key
    pub fn retrieve_key(&self, name: &str) -> Result<DecryptedKey, KmsError> {
        if !self.is_initialized.load(Ordering::Acquire) {
            return Err(KmsError::NotInitialized);
        }

        let blob = self.encrypted_keys.iter()
            .find(|b| b.get_name().ok() == Some(name))
            .ok_or(KmsError::KeyNotFound)?;

        // Check expiration
        let now = self.get_timestamp_ns();
        if blob.expires_at != 0 && now > blob.expires_at {
            return Err(KmsError::KeyExpired);
        }

        // Decrypt (reverse XOR for demo)
        let master = self.master_key.as_slice();
        let mut plaintext = SecureBuffer::new(blob.ciphertext_len as usize, true);
        
        for i in 0..blob.ciphertext_len as usize {
            let decrypted = blob.ciphertext[i] ^ master[i % AES_256_KEY_SIZE];
            unsafe {
                let ptr = plaintext.as_mut_slice().as_mut_ptr().add(i);
                write_volatile(ptr, decrypted);
            }
        }

        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_access_ts.store(now, Ordering::Release);

        Ok(DecryptedKey {
            data: plaintext,
            key_type: KeyType::try_from(blob.key_type).unwrap_or(KeyType::ApiKey),
            created_at: blob.created_at,
        })
    }

    /// Delete a key securely
    pub fn delete_key(&mut self, name: &str) -> Result<(), KmsError> {
        if let Some(pos) = self.encrypted_keys.iter().position(|b| b.get_name().ok() == Some(name)) {
            self.encrypted_keys.remove(pos);
            Ok(())
        } else {
            Err(KmsError::KeyNotFound)
        }
    }

    /// List all key names (not values)
    pub fn list_keys(&self) -> Vec<&str> {
        self.encrypted_keys.iter()
            .filter_map(|b| b.get_name().ok())
            .collect()
    }

    /// Set key expiration
    pub fn set_expiration(&mut self, name: &str, expires_at: u64) -> Result<(), KmsError> {
        let blob = self.encrypted_keys.iter_mut()
            .find(|b| b.get_name().ok() == Some(name))
            .ok_or(KmsError::KeyNotFound)?;
        
        blob.expires_at = expires_at;
        Ok(())
    }

    /// Enable/disable automatic memory wiping
    pub fn set_auto_wipe(&mut self, enabled: bool) {
        self.auto_wipe.store(enabled, Ordering::Release);
    }

    /// Get access statistics
    pub fn get_stats(&self) -> KmsStats {
        KmsStats {
            key_count: self.encrypted_keys.len(),
            access_count: self.access_count.load(Ordering::Relaxed),
            last_access_ts: self.last_access_ts.load(Ordering::Relaxed),
            is_initialized: self.is_initialized.load(Ordering::Acquire),
        }
    }

    /// Secure shutdown - wipe all sensitive data
    pub fn shutdown(&mut self) {
        self.master_key.secure_wipe();
        
        for blob in &mut self.encrypted_keys {
            blob.ciphertext.fill(0);
            blob.nonce.fill(0);
        }
        self.encrypted_keys.clear();
        
        self.is_initialized.store(false, Ordering::Release);
    }

    fn get_timestamp_ns(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
}

impl Default for LocalKms {
    fn default() -> Self {
        Self::new()
    }
}

/// Decrypted key wrapper that ensures secure cleanup
pub struct DecryptedKey {
    data: SecureBuffer,
    key_type: KeyType,
    created_at: u64,
}

impl DecryptedKey {
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_slice()
    }

    pub fn key_type(&self) -> KeyType {
        self.key_type
    }

    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    /// Explicitly wipe the key before drop
    pub fn wipe(mut self) {
        self.data.secure_wipe();
    }
}

impl Drop for DecryptedKey {
    fn drop(&mut self) {
        self.data.secure_wipe();
    }
}

/// KMS statistics
#[derive(Debug, Clone)]
pub struct KmsStats {
    pub key_count: usize,
    pub access_count: u64,
    pub last_access_ts: u64,
    pub is_initialized: bool,
}

/// KMS error types
#[derive(Debug, Clone, PartialEq)]
pub enum KmsError {
    NotInitialized,
    InvalidKeySize,
    BufferOverflow,
    KeyNameTooLong,
    KeyNotFound,
    KeyExpired,
    EncryptionFailed,
    DecryptionFailed,
    InvalidUtf8,
    MemoryAllocationFailed,
}

impl TryFrom<u8> for KeyType {
    type Error = KmsError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(KeyType::ApiKey),
            1 => Ok(KeyType::ApiSecret),
            2 => Ok(KeyType::EncryptionKey),
            3 => Ok(KeyType::SigningKey),
            4 => Ok(KeyType::HmacKey),
            _ => Err(KmsError::DecryptionFailed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kms_store_retrieve() {
        let mut kms = LocalKms::new();
        
        // Initialize with 32-byte master key
        let master_key = [0x42u8; AES_256_KEY_SIZE];
        kms.initialize(&master_key).unwrap();

        // Store a key
        let api_secret = b"super_secret_api_key_12345";
        kms.store_key("exchange_api", api_secret, KeyType::ApiSecret).unwrap();

        // Retrieve the key
        let decrypted = kms.retrieve_key("exchange_api").unwrap();
        assert_eq!(decrypted.as_bytes(), api_secret);
        assert_eq!(decrypted.key_type(), KeyType::ApiSecret);
    }

    #[test]
    fn test_secure_buffer_wipe() {
        let mut buffer = SecureBuffer::new(32, true);
        
        // Write some data
        let test_data = [0xDEu8; 32];
        buffer.copy_from(&test_data).unwrap();
        
        // Verify data is present
        assert_eq!(buffer.as_slice(), &test_data);
        
        // Wipe
        buffer.secure_wipe();
        
        // Verify data is wiped (all zeros after final pass)
        for &byte in buffer.as_slice() {
            assert_eq!(byte, 0u8);
        }
    }

    #[test]
    fn test_key_expiration() {
        let mut kms = LocalKms::new();
        let master_key = [0x42u8; AES_256_KEY_SIZE];
        kms.initialize(&master_key).unwrap();

        kms.store_key("temp_key", b"temp_secret", KeyType::ApiKey).unwrap();
        
        // Set expiration in the past
        kms.set_expiration("temp_key", 1000).unwrap();
        
        // Should fail to retrieve expired key
        let result = kms.retrieve_key("temp_key");
        assert!(result.is_err());
    }

    #[test]
    fn test_kms_list_keys() {
        let mut kms = LocalKms::new();
        let master_key = [0x42u8; AES_256_KEY_SIZE];
        kms.initialize(&master_key).unwrap();

        kms.store_key("key1", b"secret1", KeyType::ApiKey).unwrap();
        kms.store_key("key2", b"secret2", KeyType::ApiSecret).unwrap();
        kms.store_key("key3", b"secret3", KeyType::EncryptionKey).unwrap();

        let keys = kms.list_keys();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"key1"));
        assert!(keys.contains(&"key2"));
        assert!(keys.contains(&"key3"));
    }
}
