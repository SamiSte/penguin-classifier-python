/* Keep every +/- click synchronous with the input's current browser value. */
window.penguinUi = {
    adjustValue: function (value, step, fallback, direction) {
        const parsed = value === null || value === "" ? NaN : Number(value);
        const current = Number.isFinite(parsed) ? parsed : fallback;
        // Keep manually entered precision; only remove floating point artefacts.
        return Math.max(step, Math.round((current + direction * step) * 1000000) / 1000000);
    }
};
