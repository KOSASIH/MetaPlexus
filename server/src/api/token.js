import express from 'express';
import jwt from 'jsonwebtoken';

const router = express.Router();

router.post('/refresh', async (req, res) => {
  try {
    const token = req.header('Authorization');
    if (!token) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
    const decodedToken = jwt.verify(token, process.env.SECRET_KEY);
    const newToken = jwt.sign({ userId: decodedToken.userId }, process.env.SECRET_KEY, {
      expiresIn: '1h',
    });
    res.status(200).json({ token: newToken });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

export default router;
