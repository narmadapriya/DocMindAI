import { create } from "zustand";
import { authService, } from "../services/auth";
import { getErrorMessage } from "../services/api";
const ACCESS_TOKEN_KEY = "docmind_token";
const REFRESH_TOKEN_KEY = "docmind_refresh_token";
const USER_KEY = "docmind_user";
function storedUser() {
    const raw = localStorage.getItem(USER_KEY);
    if (!raw)
        return null;
    try {
        return JSON.parse(raw);
    }
    catch {
        localStorage.removeItem(USER_KEY);
        return null;
    }
}
function clearSession() {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
}
export const useAuthStore = create((set, get) => ({
    user: storedUser(),
    token: localStorage.getItem(ACCESS_TOKEN_KEY),
    refreshToken: localStorage.getItem(REFRESH_TOKEN_KEY),
    loading: false,
    initialized: false,
    error: null,
    initialize: async () => {
        if (get().initialized)
            return;
        if (!get().token) {
            set({ initialized: true, user: null });
            return;
        }
        try {
            const user = await authService.me();
            localStorage.setItem(USER_KEY, JSON.stringify(user));
            set({ user, initialized: true, error: null });
        }
        catch {
            clearSession();
            set({
                user: null,
                token: null,
                refreshToken: null,
                initialized: true,
            });
        }
    },
    login: async (payload) => {
        set({ loading: true, error: null });
        try {
            const tokens = await authService.login(payload);
            localStorage.setItem(ACCESS_TOKEN_KEY, tokens.access_token);
            if (tokens.refresh_token) {
                localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refresh_token);
            }
            else {
                localStorage.removeItem(REFRESH_TOKEN_KEY);
            }
            set({
                token: tokens.access_token,
                refreshToken: tokens.refresh_token ?? null,
            });
            const user = await authService.me();
            localStorage.setItem(USER_KEY, JSON.stringify(user));
            set({
                user,
                initialized: true,
                error: null,
            });
        }
        catch (error) {
            clearSession();
            set({
                user: null,
                token: null,
                refreshToken: null,
                error: getErrorMessage(error),
            });
            throw error;
        }
        finally {
            set({ loading: false });
        }
    },
    register: async (payload) => {
        set({ loading: true, error: null });
        try {
            await authService.register(payload);
        }
        catch (error) {
            set({ error: getErrorMessage(error) });
            throw error;
        }
        finally {
            set({ loading: false });
        }
    },
    logout: () => {
        clearSession();
        set({
            user: null,
            token: null,
            refreshToken: null,
            initialized: true,
            error: null,
        });
    },
    updateProfile: async (payload) => {
        set({ loading: true, error: null });
        try {
            const user = await authService.updateProfile(payload);
            localStorage.setItem(USER_KEY, JSON.stringify(user));
            set({ user });
        }
        catch (error) {
            set({ error: getErrorMessage(error) });
            throw error;
        }
        finally {
            set({ loading: false });
        }
    },
}));
