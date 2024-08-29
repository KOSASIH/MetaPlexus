import jwt from 'jsonwebtoken';
import { SECRET_KEY } from '../config';

class TokenService {
  async generateToken(userId) {
    return jwt.sign({ userId }, SECRET_KEY, {
      expiresIn: '1h',
    });
  }

  async verifyToken(token) {
    try {
      const decodedToken = jwt.verify(token, SECRET_KEY);
      return decodedToken.userId;
    } catch (error) {
      throw new Error('Invalid token');
    }
  }

  async refreshToken(token) {
    try {
      const decodedToken = jwt.verify(token, SECRET_KEY);
      const newToken = await this.generateToken(decodedToken.userId);
      return newToken;
    } catch (error) {
      throw new Error('Invalid token');
    }
  }

  async revokeToken(token) {
    // Implement token blacklisting or revocation logic here
    // For example, you can store revoked tokens in a database or cache
    // and check against it in the verifyToken method
  }
}

export default new TokenService();
