import java.math.BigInteger;
import java.util.Random;

public class Keys {

	private BigInteger[]	publicKey;
	private BigInteger[]	privateKey;

	public BigInteger[] getPublicKey() {
		return publicKey;
	}

	public BigInteger[] getPrivateKey() {
		return privateKey;
	}

	public Keys() {
		// init public key [encryption exponent, modulus]
		publicKey = new BigInteger[2];

		// init private key [decryption exponent, modulus]
		privateKey = new BigInteger[2];

		// init param
		int maxVal = (int) Math.pow( 2, 100 );
		int minVal = (int) Math.pow( 1, 100 );
		BigInteger max = new BigInteger( Integer.toString( maxVal ) );
		Random random = new Random();
		int maxLength = max.bitLength();
		int certainty = 40;
		BigInteger one = new BigInteger( "1" );

		// generating big primes
		BigInteger p = Prime.getPrime( minVal, maxVal, certainty );
		BigInteger q = Prime.getPrime( minVal, maxVal, certainty );

		// modulus
		BigInteger n = p.multiply( q );

		// Euler's totient function
		BigInteger phi = ( p.subtract( one ) ).multiply( q.subtract( one ) );

		// the encryption exponent
		BigInteger e;
		do {
			e = new BigInteger( maxLength, random );
		} while ( ( e.gcd( phi ) ).intValue() != 1 );

		// the decryption exponent
		BigInteger d = e.modInverse( phi );

		// assigning values to public and private keys
		this.publicKey[0] = e;
		this.publicKey[1] = n;

		this.privateKey[0] = d;
		this.privateKey[1] = n;

	}
}
