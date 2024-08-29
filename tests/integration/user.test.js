import { expect } from '@jest/globals';
import { beforeEach, afterEach, describe, it } from '@jest/globals';
import { User } from '../server/src/db/models/User';
import { api } from '../server/src/api/user';
import { auth } from '../server/src/auth/auth';
import { tokenService } from '../server/src/services/token';
import { app } from '../server/src/app';

describe('User Integration Tests', () => {
  let user;
  let token;

  beforeEach(async () => {
    // Create a test user
    user = await User.create({
      email: 'test@example.com',
      password: 'password123',
      name: 'John Doe',
    });

    // Generate a token for the test user
    token = await tokenService.generateToken(user);
  });

  afterEach(async () => {
    // Clean up the test user
    await user.destroy();
  });

  describe('GET /api/users/me', () => {
    it('should return the current user', async () => {
      const response = await api.get('/api/users/me', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      expect(response.status).toBe(200);
      expect(response.data).toEqual({
        id: user.id,
        email: user.email,
        name: user.name,
      });
    });
  });

  describe('POST /api/users', () => {
    it('should create a new user', async () => {
      const response = await api.post('/api/users', {
        email: 'new@example.com',
        password: 'password123',
        name: 'Jane Doe',
      });

      expect(response.status).toBe(201);
      expect(response.data).toEqual({
        id: expect.any(Number),
        email: 'new@example.com',
        name: 'Jane Doe',
      });
    });
  });

  describe('PUT /api/users/:id', () => {
    it('should update an existing user', async () => {
      const response = await api.put(`/api/users/${user.id}`, {
        name: 'Updated Name',
      }, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      expect(response.status).toBe(200);
      expect(response.data).toEqual({
        id: user.id,
        email: user.email,
        name: 'Updated Name',
      });
    });
  });

  describe('DELETE /api/users/:id', () => {
    it('should delete an existing user', async () => {
      const response = await api.delete(`/api/users/${user.id}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      expect(response.status).toBe(204);
    });
  });
});
