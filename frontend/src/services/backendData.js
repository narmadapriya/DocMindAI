import { api } from "./api";

export const backendDataService = {
  async dashboard(days = 7) {
    const { data } = await api.get(
      `/api/analytics/dashboard?days=${days}`,
    );
    return data;
  },

  async queriesOverTime(days = 7) {
    const { data } = await api.get(
      `/api/analytics/queries-over-time?days=${days}`,
    );
    return data;
  },
};
