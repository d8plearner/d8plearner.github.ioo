import java.math.BigInteger;

public class RSA {

	// public key should respect the format [encryption exponent, modulus]
	public static String encrypt( BigInteger[] publicKey, String message ) {
		// encryption exponent
		BigInteger e = publicKey[0];

		// modulus
		BigInteger n = publicKey[1];

		BigInteger[] ascii = Utilities.toAscii( message );
		String encryptedMessage = "";

		for ( BigInteger a : ascii ) {
			encryptedMessage += ( a.modPow( e, n ) ).longValue() + " ";
		}
		return encryptedMessage;
	}

	// private key should respect the format [decryption exponent, modulus]
	public static String decrypt( BigInteger[] privateKey, String encryptedMessage ) {
		// decryption exponent
		BigInteger d = privateKey[0];

		// modulus
		BigInteger n = privateKey[1];

		String decryptedMessage = "";

		// we split the encrypted message into blocks
		for ( String block : encryptedMessage.split( " " ) ) {
			BigInteger cipheredBlock = new BigInteger( block );
			decryptedMessage += (char) ( cipheredBlock.modPow( d, n ) ).intValue();
		}

		return decryptedMessage;
	}
}
