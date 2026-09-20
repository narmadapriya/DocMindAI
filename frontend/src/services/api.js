import axios from "axios";
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim().replace(/\/$/, "") || "";
export const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 600_000,
});
api.interceptors.request.use((config) => {
    const token = localStorage.getItem("docmind_token");
    if (token)
        config.headers.Authorization = `Bearer ${token}`;
    return config;
});
api.interceptors.response.use((response) => response, (error) => {
    if (error.response?.status === 401) {
        localStorage.removeItem("docmind_token");
        localStorage.removeItem("docmind_user");
        if (!window.location.pathname.startsWith("/login")) {
            window.dispatchEvent(new CustomEvent("docmind:unauthorized"));
        }
    }
    return Promise.reject(error);
});
export function getErrorMessage(error) {
    if (axios.isAxiosError(error)) {
        const data = error.response?.data;
        return data?.error?.message || data?.detail || data?.message || error.message || "Request failed.";
    }
    return error instanceof Error ? error.message : "Unexpected error.";
}
export async function requestWithFallback(paths, config) {
    let lastError;
    const enableFallbacks = String(import.meta.env.VITE_ENABLE_API_FALLBACKS ?? "true") !== "false";
    for (const [index, path] of paths.entries()) {
        try {
            return await api.request({ ...config, url: path });
        }
        catch (error) {
            lastError = error;
            if (!enableFallbacks || index === paths.length - 1 || !axios.isAxiosError(error))
                break;
            if (![404, 405].includes(error.response?.status || 0))
                break;
        }
    }
    throw lastError;
}
