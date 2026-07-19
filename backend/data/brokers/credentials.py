"""
================================================================================
QUANT X - BROKER CREDENTIALS ENCRYPTION
================================================================================
Secure encryption/decryption for broker credentials using Fernet (AES-128-CBC).
================================================================================
"""

import os
import json
import base64
import logging
from typing import Dict, Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# ============================================================================
# ENCRYPTION KEY MANAGEMENT
# ============================================================================


def _get_encryption_key() -> bytes:
    """
    Get or generate encryption key from environment.
    Uses BROKER_ENCRYPTION_KEY if set, otherwise derives from SECRET_KEY.
    """
    # Check for dedicated encryption key
    encryption_key = os.getenv("BROKER_ENCRYPTION_KEY")

    if encryption_key:
        # Validate it's a valid Fernet key
        try:
            Fernet(encryption_key.encode())
            return encryption_key.encode()
        except Exception:
            logger.warning("Invalid BROKER_ENCRYPTION_KEY, deriving from SECRET_KEY")

    # Derive key from SECRET_KEY — refuse to use default in production
    secret_key = os.getenv("SECRET_KEY", "")
    if not secret_key:
        raise RuntimeError("SECRET_KEY must be set for broker credential encryption")
    # Salt must be set explicitly in production. A hardcoded default salt
    # weakens the KDF (predictable key material), so fail closed rather than
    # silently fall back — mirrors config.validate_startup's production posture.
    salt_env = os.getenv("ENCRYPTION_SALT")
    if not salt_env:
        if os.getenv("APP_ENV", "development") == "production":
            raise RuntimeError(
                "ENCRYPTION_SALT must be set in production for broker credential encryption"
            )
        logger.warning(
            "ENCRYPTION_SALT not set — using insecure development default; "
            "set ENCRYPTION_SALT before deploying to production"
        )
        salt_env = "quantx-broker-salt"
    salt = salt_env.encode()

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )

    key = base64.urlsafe_b64encode(kdf.derive(secret_key.encode()))
    return key


def _get_fernet() -> Fernet:
    """Get Fernet instance with encryption key"""
    return Fernet(_get_encryption_key())


# ============================================================================
# ENCRYPTION/DECRYPTION FUNCTIONS
# ============================================================================

def encrypt_credentials(credentials: Dict[str, Any]) -> str:
    """
    Encrypt broker credentials dictionary.

    Args:
        credentials: Dictionary containing API keys, tokens, etc.

    Returns:
        Base64-encoded encrypted string

    Raises:
        RuntimeError: when key material is unavailable (config issue —
            propagate so ops can see "SECRET_KEY must be set" instead
            of a generic "Failed to encrypt").
        ValueError: on actual encryption errors.
    """
    try:
        fernet = _get_fernet()

        # Convert to JSON and encode
        json_data = json.dumps(credentials)
        encrypted = fernet.encrypt(json_data.encode())

        return encrypted.decode()

    except RuntimeError:
        # Config error — let it surface verbatim.
        raise
    except Exception as e:
        logger.error(f"Error encrypting credentials: {e}")
        raise ValueError("Failed to encrypt credentials")


def decrypt_credentials(encrypted_data: str) -> Dict[str, Any]:
    """
    Decrypt broker credentials.

    Args:
        encrypted_data: Base64-encoded encrypted string

    Returns:
        Decrypted credentials dictionary
    """
    try:
        fernet = _get_fernet()

        # Decrypt and parse JSON
        decrypted = fernet.decrypt(encrypted_data.encode())
        credentials = json.loads(decrypted.decode())

        return credentials

    except InvalidToken:
        logger.error("Invalid token - credentials may be corrupted or key changed")
        raise ValueError("Failed to decrypt credentials - invalid token")
    except Exception as e:
        logger.error(f"Error decrypting credentials: {e}")
        raise ValueError("Failed to decrypt credentials")


def generate_encryption_key() -> str:
    """
    Generate a new Fernet encryption key.
    Use this to generate BROKER_ENCRYPTION_KEY for production.

    Returns:
        Base64-encoded Fernet key
    """
    return Fernet.generate_key().decode()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def mask_credentials(credentials: Dict[str, Any]) -> Dict[str, str]:
    """
    Mask sensitive credential values for logging/display.

    Args:
        credentials: Original credentials dict

    Returns:
        Masked credentials dict
    """
    masked = {}

    for key, value in credentials.items():
        if isinstance(value, str) and len(value) > 4:
            masked[key] = value[:4] + "*" * (len(value) - 4)
        else:
            masked[key] = "****"

    return masked


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Generate a new key
    print(f"New encryption key: {generate_encryption_key()}")

    # Test encryption/decryption
    test_creds = {
        "api_key": "test_api_key_12345",
        "api_secret": "test_secret_67890",
        "access_token": "jwt_token_here"
    }

    encrypted = encrypt_credentials(test_creds)
    print(f"Encrypted: {encrypted[:50]}...")

    decrypted = decrypt_credentials(encrypted)
    print(f"Decrypted: {decrypted}")

    print(f"Masked: {mask_credentials(test_creds)}")
