import { requestWithFallback } from "./api";
function arrayFrom(data, keys) {
    if (Array.isArray(data))
        return data;
    const record = (data || {});
    for (const key of keys)
        if (Array.isArray(record[key]))
            return record[key];
    return [];
}
export const metaService = {
    async citations() {
        const { data } = await requestWithFallback(["/api/citations", "/api/citations/"], { method: "GET" });
        return arrayFrom(data, ["citations", "items"]);
    },
    async analytics() {
        const { data } = await requestWithFallback(["/api/analytics", "/api/analytics/"], { method: "GET" });
        return arrayFrom(data, ["analytics", "events", "items"]);
    },
    async settings() {
        const { data } = await requestWithFallback(["/api/settings", "/api/settings/"], { method: "GET" });
        return data;
    },
    async updateSettings(payload) {
        const { data } = await requestWithFallback(["/api/settings", "/api/settings/"], { method: "PUT", data: payload });
        return data;
    },
    async chats() {
        const { data } = await requestWithFallback(["/api/chats", "/api/chat/history"], { method: "GET" });
        return arrayFrom(data, ["chats", "items", "conversations"]);
    },
};
