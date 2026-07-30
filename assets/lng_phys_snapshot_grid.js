var dagfuncs = window.dashAgGridFunctions =
    window.dashAgGridFunctions || {};

dagfuncs.physSnapshotOneDecimal = function (params) {
    if (
        params.value === null ||
        params.value === undefined ||
        params.value === ""
    ) {
        return "—";
    }

    var numericValue = Number(params.value);
    if (!Number.isFinite(numericValue)) {
        return "—";
    }

    return numericValue.toLocaleString("en-US", {
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
    });
};

dagfuncs.physSnapshotCountryIsContinuation = function (params) {
    if (
        !params ||
        !params.node ||
        !params.api ||
        !Number.isInteger(params.node.rowIndex) ||
        params.node.rowIndex <= 0
    ) {
        return false;
    }

    var previousNode = params.api.getDisplayedRowAtIndex(
        params.node.rowIndex - 1
    );
    return Boolean(
        previousNode &&
        previousNode.data &&
        previousNode.data.Country === params.value
    );
};

dagfuncs.physSnapshotCountryGroupLabel = function (params) {
    if (!params || params.value === null || params.value === undefined) {
        return "";
    }
    return dagfuncs.physSnapshotCountryIsContinuation(params)
        ? ""
        : String(params.value);
};
