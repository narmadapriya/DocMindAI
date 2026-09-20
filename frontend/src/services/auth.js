import { api, requestWithFallback } from "./api";
export const authService = {
    async login(payload) {
        const email = payload.email.trim().toLowerCase();
        const password = payload.password;
        const form = new URLSearchParams();
        form.append("username", email);
        form.append("password", password);
        const { data } = await api.post("/api/auth/login", form.toString(), {
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
            },
        });
        return data;
    },
    async register(payload) {
        const { data } = await api.post("/api/auth/register", {
            username: payload.username.trim(),
            email: payload.email.trim().toLowerCase(),
            password: payload.password,
        });
        return data;
    },
    async me() {
        const { data } = await api.get("/api/users/me");
        return data;
    },
    async refresh(refreshToken) {
        const { data } = await api.post("/api/auth/refresh", { refresh_token: refreshToken });
        return data;
    },
    async logout() {
        await api.post("/api/auth/logout");
    },
    async requestPasswordReset(email) {
        const { data } = await api.post("/api/auth/password-reset/request", {
            email: email.trim().toLowerCase(),
        });
        return data;
    },
    async resetPassword(payload) {
        const { data } = await api.post("/api/auth/password-reset/confirm", {
            token: payload.token,
            new_password: payload.newPassword,
        });
        return data;
    },
    async updateProfile(payload) {
        const { data } = await requestWithFallback(["/api/users/me", "/api/users/profile"], {
            method: "PUT",
            data: {
                ...payload,
                ...(payload.username !== undefined
                    ? { username: payload.username.trim() }
                    : {}),
                ...(payload.email !== undefined
                    ? { email: payload.email.trim().toLowerCase() }
                    : {}),
            },
        });
        return data;
    },
};
