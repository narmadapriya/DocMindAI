import { api } from "./api";

export const ragService = {
  async chat(payload) {
    const { data } = await api.post(
      "/api/v1/chat",
      payload,
    );
    return data;
  },

  async summary(payload) {
    const { data } = await api.post(
      "/api/v1/summary",
      payload,
    );
    return data;
  },

  async compare(payload) {
    const { data } = await api.post(
      "/api/v1/compare",
      payload,
    );
    return data;
  },
};
