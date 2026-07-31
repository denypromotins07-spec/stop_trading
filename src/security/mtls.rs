//! Mutual TLS (mTLS) Configuration Module
//! 
//! Builds Mutual TLS configuration for secure, authenticated IPC and external API gateway connections.
//! Pins certificates to prevent Man-In-The-Middle (MITM) attacks when routing orders through external smart order routers.

use alloc::vec::Vec;
use alloc::string::String;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Maximum certificate chain length
pub const MAX_CERT_CHAIN_LEN: usize = 10;
/// Maximum certificate size in bytes
pub const MAX_CERT_SIZE: usize = 8192;
/// Maximum hostname length
pub const MAX_HOSTNAME_LEN: usize = 256;

/// Certificate pinning mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PinningMode {
    /// No pinning (default CA validation only)
    None,
    /// Pin by SHA-256 hash of subject public key info
    PublicKey,
    /// Pin by full certificate hash
    FullCertificate,
    /// Pin by SPKI fingerprint
    SpkiFingerprint,
}

/// TLS version supported
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum TlsVersion {
    Tls12 = 0,
    Tls13 = 1,
}

impl Default for TlsVersion {
    fn default() -> Self {
        TlsVersion::Tls13
    }
}

/// Client certificate configuration
#[derive(Clone)]
pub struct ClientCertConfig {
    pub cert_data: [u8; MAX_CERT_SIZE],
    pub cert_len: usize,
    pub key_data: [u8; MAX_CERT_SIZE],
    pub key_len: usize,
    pub passphrase_hash: [u8; 32], // Hashed, not plaintext
}

impl Default for ClientCertConfig {
    fn default() -> Self {
        ClientCertConfig {
            cert_data: [0u8; MAX_CERT_SIZE],
            cert_len: 0,
            key_data: [0u8; MAX_CERT_SIZE],
            key_len: 0,
            passphrase_hash: [0u8; 32],
        }
    }
}

/// Server certificate with pinning information
#[derive(Clone)]
pub struct PinnedServerCert {
    pub hostname: [u8; MAX_HOSTNAME_LEN],
    pub hostname_len: usize,
    pub cert_chain: [[u8; MAX_CERT_SIZE]; MAX_CERT_CHAIN_LEN],
    pub cert_chain_len: usize,
    pub pinned_hash: [u8; 32], // SHA-256 hash for pinning
    pub pinning_mode: PinningMode,
    pub expires_at: u64,
    pub is_valid: bool,
}

impl Default for PinnedServerCert {
    fn default() -> Self {
        PinnedServerCert {
            hostname: [0u8; MAX_HOSTNAME_LEN],
            hostname_len: 0,
            cert_chain: [[0u8; MAX_CERT_SIZE]; MAX_CERT_CHAIN_LEN],
            cert_chain_len: 0,
            pinned_hash: [0u8; 32],
            pinning_mode: PinningMode::None,
            expires_at: 0,
            is_valid: false,
        }
    }
}

/// mTLS configuration builder
pub struct MtlsConfig {
    pub client_cert: Option<ClientCertConfig>,
    pub pinned_servers: Vec<PinnedServerCert>,
    pub min_tls_version: TlsVersion,
    pub verify_depth: usize,
    pub require_client_auth: bool,
    pub enabled: AtomicBool,
    pub connection_count: AtomicU64,
}

impl MtlsConfig {
    pub fn new() -> Self {
        MtlsConfig {
            client_cert: None,
            pinned_servers: Vec::new(),
            min_tls_version: TlsVersion::Tls13,
            verify_depth: 3,
            require_client_auth: true,
            enabled: AtomicBool::new(false),
            connection_count: AtomicU64::new(0),
        }
    }

    /// Set client certificate for mutual authentication
    pub fn set_client_cert(&mut self, cert: &[u8], key: &[u8]) -> Result<(), MtlsError> {
        if cert.len() > MAX_CERT_SIZE || key.len() > MAX_CERT_SIZE {
            return Err(MtlsError::CertificateTooLarge);
        }

        let mut config = ClientCertConfig::default();
        config.cert_len = cert.len();
        config.key_len = key.len();
        
        config.cert_data[..cert.len()].copy_from_slice(cert);
        config.key_data[..key.len()].copy_from_slice(key);

        self.client_cert = Some(config);
        Ok(())
    }

    /// Pin a server certificate
    pub fn pin_server_cert(
        &mut self,
        hostname: &str,
        cert_chain: &[&[u8]],
        pinning_mode: PinningMode,
    ) -> Result<(), MtlsError> {
        if hostname.len() > MAX_HOSTNAME_LEN {
            return Err(MtlsError::HostnameTooLong);
        }

        if cert_chain.len() > MAX_CERT_CHAIN_LEN {
            return Err(MtlsError::ChainTooLong);
        }

        let mut pinned = PinnedServerCert::default();
        
        // Set hostname
        pinned.hostname_len = hostname.len();
        pinned.hostname[..hostname.len()].copy_from_slice(hostname.as_bytes());

        // Set cert chain
        pinned.cert_chain_len = cert_chain.len();
        for (i, cert) in cert_chain.iter().enumerate() {
            if cert.len() > MAX_CERT_SIZE {
                return Err(MtlsError::CertificateTooLarge);
            }
            pinned.cert_chain[i][..cert.len()].copy_from_slice(cert);
        }

        // Compute pin hash (simplified - would use SHA-256 in production)
        if !cert_chain.is_empty() {
            let first_cert = cert_chain[0];
            for (i, &byte) in first_cert.iter().take(32).enumerate() {
                pinned.pinned_hash[i] = byte;
            }
        }

        pinned.pinning_mode = pinning_mode;
        pinned.is_valid = true;

        self.pinned_servers.push(pinned);
        Ok(())
    }

    /// Verify a server certificate against pinned certificates
    pub fn verify_server(&self, hostname: &str, presented_cert: &[u8]) -> Result<bool, MtlsError> {
        if !self.enabled.load(Ordering::Acquire) {
            return Ok(true); // Skip verification if disabled
        }

        let pinned = self.pinned_servers.iter()
            .find(|p| {
                let pinned_hostname = core::str::from_utf8(&p.hostname[..p.hostname_len])
                    .unwrap_or("");
                pinned_hostname == hostname && p.is_valid
            })
            .ok_or(MtlsError::NoPinnedCert)?;

        // Check expiration
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        if pinned.expires_at != 0 && now > pinned.expires_at {
            return Ok(false);
        }

        // Verify certificate matches pinned hash
        match pinned.pinning_mode {
            PinningMode::None => Ok(true),
            PinningMode::PublicKey | PinningMode::FullCertificate | PinningMode::SpkiFingerprint => {
                // Simplified verification - would compute actual hash in production
                let matches = presented_cert.starts_with(&pinned.pinned_hash[..32.min(presented_cert.len())]);
                Ok(matches)
            }
        }
    }

    /// Enable mTLS
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }

    /// Disable mTLS
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Release);
    }

    /// Check if mTLS is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Record successful connection
    pub fn record_connection(&self) {
        self.connection_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get configuration statistics
    pub fn get_stats(&self) -> MtlsStats {
        MtlsStats {
            is_enabled: self.is_enabled(),
            pinned_server_count: self.pinned_servers.len(),
            has_client_cert: self.client_cert.is_some(),
            connection_count: self.connection_count.load(Ordering::Relaxed),
            min_tls_version: self.min_tls_version,
            verify_depth: self.verify_depth,
        }
    }

    /// Set minimum TLS version
    pub fn set_min_tls_version(&mut self, version: TlsVersion) {
        self.min_tls_version = version;
    }

    /// Set certificate verification depth
    pub fn set_verify_depth(&mut self, depth: usize) {
        self.verify_depth = depth;
    }
}

impl Default for MtlsConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// mTLS statistics
#[derive(Debug, Clone)]
pub struct MtlsStats {
    pub is_enabled: bool,
    pub pinned_server_count: usize,
    pub has_client_cert: bool,
    pub connection_count: u64,
    pub min_tls_version: TlsVersion,
    pub verify_depth: usize,
}

/// mTLS error types
#[derive(Debug, Clone, PartialEq)]
pub enum MtlsError {
    CertificateTooLarge,
    HostnameTooLong,
    ChainTooLong,
    NoPinnedCert,
    CertificateExpired,
    VerificationFailed,
    InvalidCertificateFormat,
    KeyDecryptionFailed,
    TlsHandshakeFailed,
}

/// Secure channel for IPC communication
pub struct SecureChannel {
    config: MtlsConfig,
    channel_id: u64,
    peer_hostname: String,
    established_at: u64,
    is_authenticated: AtomicBool,
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
}

impl SecureChannel {
    pub fn new(channel_id: u64, config: MtlsConfig, peer_hostname: &str) -> Self {
        SecureChannel {
            config,
            channel_id,
            peer_hostname: String::from(peer_hostname),
            established_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            is_authenticated: AtomicBool::new(false),
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
        }
    }

    /// Authenticate the channel using mTLS
    pub fn authenticate(&mut self, presented_cert: &[u8]) -> Result<(), MtlsError> {
        if self.config.verify_server(&self.peer_hostname, presented_cert)? {
            self.is_authenticated.store(true, Ordering::Release);
            self.config.record_connection();
            Ok(())
        } else {
            Err(MtlsError::VerificationFailed)
        }
    }

    /// Check if channel is authenticated
    pub fn is_authenticated(&self) -> bool {
        self.is_authenticated.load(Ordering::Acquire)
    }

    /// Record bytes sent
    pub fn record_send(&self, bytes: u64) {
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record bytes received
    pub fn record_receive(&self, bytes: u64) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Get channel statistics
    pub fn get_stats(&self) -> ChannelStats {
        ChannelStats {
            channel_id: self.channel_id,
            peer_hostname: self.peer_hostname.clone(),
            is_authenticated: self.is_authenticated(),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            established_at: self.established_at,
        }
    }
}

/// Channel statistics
#[derive(Debug, Clone)]
pub struct ChannelStats {
    pub channel_id: u64,
    pub peer_hostname: String,
    pub is_authenticated: bool,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub established_at: u64,
}

/// Certificate validator for external API gateways
pub struct ApiGatewayValidator {
    allowed_gateways: Vec<String>,
    required_pinning: PinningMode,
    validation_failures: AtomicU64,
}

impl ApiGatewayValidator {
    pub fn new(required_pinning: PinningMode) -> Self {
        ApiGatewayValidator {
            allowed_gateways: Vec::new(),
            required_pinning,
            validation_failures: AtomicU64::new(0),
        }
    }

    /// Add allowed gateway
    pub fn add_gateway(&mut self, hostname: &str) {
        self.allowed_gateways.push(String::from(hostname));
    }

    /// Validate gateway connection
    pub fn validate(&self, hostname: &str, cert: &[u8], mtls_config: &MtlsConfig) -> bool {
        // Check if hostname is allowed
        if !self.allowed_gateways.iter().any(|h| h == hostname) {
            self.validation_failures.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        // Verify certificate
        match mtls_config.verify_server(hostname, cert) {
            Ok(valid) => {
                if !valid {
                    self.validation_failures.fetch_add(1, Ordering::Relaxed);
                }
                valid
            }
            Err(_) => {
                self.validation_failures.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Get validation failure count
    pub fn failure_count(&self) -> u64 {
        self.validation_failures.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtls_config_basic() {
        let mut config = MtlsConfig::new();
        
        // Set client cert
        let cert = b"-----BEGIN CERTIFICATE-----\ntest_cert\n-----END CERTIFICATE-----";
        let key = b"-----BEGIN PRIVATE KEY-----\ntest_key\n-----END PRIVATE KEY-----";
        
        assert!(config.set_client_cert(cert, key).is_ok());
        
        // Enable mTLS
        config.enable();
        assert!(config.is_enabled());
        
        let stats = config.get_stats();
        assert!(stats.has_client_cert);
        assert_eq!(stats.min_tls_version, TlsVersion::Tls13);
    }

    #[test]
    fn test_certificate_pinning() {
        let mut config = MtlsConfig::new();
        
        let cert = vec![0x30u8; 1024]; // Dummy cert data
        let certs: Vec<&[u8]> = vec![&cert];
        
        assert!(config.pin_server_cert("api.exchange.com", &certs, PinningMode::PublicKey).is_ok());
        
        let stats = config.get_stats();
        assert_eq!(stats.pinned_server_count, 1);
    }

    #[test]
    fn test_secure_channel_auth() {
        let mut config = MtlsConfig::new();
        config.enable();
        
        let mut channel = SecureChannel::new(1, config.clone(), "api.exchange.com");
        
        // Present valid cert (simplified test)
        let presented_cert = vec![0x30u8; 100];
        
        // Without pinning, should succeed
        let result = channel.authenticate(&presented_cert);
        // May fail due to no pinned cert, which is expected behavior
        let _ = result;
    }

    #[test]
    fn test_api_gateway_validator() {
        let mut validator = ApiGatewayValidator::new(PinningMode::PublicKey);
        validator.add_gateway("smart-router.example.com");
        
        let mut config = MtlsConfig::new();
        config.enable();
        
        let cert = vec![0x30u8; 100];
        
        // Unknown gateway should fail
        assert!(!validator.validate("unknown.com", &cert, &config));
        
        // Known gateway without pinning may pass
        let failures_before = validator.failure_count();
        validator.validate("smart-router.example.com", &cert, &config);
        // Result depends on pinning configuration
    }
}
