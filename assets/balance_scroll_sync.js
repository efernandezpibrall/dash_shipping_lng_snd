(function () {
  const TABLE_GROUPS = [
    ["balance-woodmac-table", "balance-comparison-delta-table"],
    [
      "capacity-page-woodmac-table",
      "capacity-page-ea-table",
      "capacity-page-internal-scenario-table",
    ],
  ];

  const state = (window.__balanceTableSyncState =
    window.__balanceTableSyncState || {
      bindings: {},
      observer: null,
      rafId: null,
    });

  function getScrollableElements(tableId) {
    const root = document.getElementById(tableId);
    if (!root) {
      return [];
    }

    const selectorPriority = [
      ".dash-spreadsheet-container",
      ".dt-table-container__row-1",
      ".dash-spreadsheet-inner",
      ".dash-table-container",
    ];

    const scrollableElements = [];
    const seenElements = new Set();

    for (const selector of selectorPriority) {
      const candidates = [];
      if (root.matches && root.matches(selector)) {
        candidates.push(root);
      }
      candidates.push(...root.querySelectorAll(selector));

      candidates.forEach(function (element) {
        if (seenElements.has(element)) {
          return;
        }

        if (
          element.scrollHeight > element.clientHeight + 1 ||
          element.scrollWidth > element.clientWidth + 1
        ) {
          seenElements.add(element);
          scrollableElements.push(element);
        }
      });
    }

    return scrollableElements;
  }

  function cleanupBinding(bindingKey) {
    const existingBinding = state.bindings[bindingKey];
    if (!existingBinding) {
      return;
    }

    existingBinding.listeners.forEach(function (listener) {
      listener.element.removeEventListener("scroll", listener.handler);
    });
    delete state.bindings[bindingKey];
  }

  function areScrollableMapsEqual(currentMap, nextMap, tableIds) {
    return tableIds.every(function (tableId) {
      const currentElements = currentMap[tableId] || [];
      const nextElements = nextMap[tableId] || [];

      return (
        currentElements.length === nextElements.length &&
        currentElements.every(function (element, index) {
          return element === nextElements[index];
        })
      );
    });
  }

  function bindScrollSyncGroup(tableIds) {
    const scrollablesById = {};
    const bindingKey = tableIds.join("::");
    const currentBinding = state.bindings[bindingKey];

    tableIds.forEach(function (tableId) {
      scrollablesById[tableId] = getScrollableElements(tableId);
    });

    if (
      tableIds.some(function (tableId) {
        return !scrollablesById[tableId].length;
      })
    ) {
      cleanupBinding(bindingKey);
      return;
    }

    if (
      currentBinding &&
      areScrollableMapsEqual(
        currentBinding.scrollablesById,
        scrollablesById,
        tableIds
      )
    ) {
      return;
    }

    cleanupBinding(bindingKey);

    let isSyncing = false;

    const syncScrollPosition = function (sourceElement, sourceTableId) {
      if (isSyncing) {
        return;
      }

      isSyncing = true;
      tableIds.forEach(function (tableId) {
        if (tableId === sourceTableId) {
          return;
        }

        (scrollablesById[tableId] || []).forEach(function (targetElement) {
          if (targetElement === sourceElement) {
            return;
          }

          targetElement.scrollTop = sourceElement.scrollTop;
          targetElement.scrollLeft = sourceElement.scrollLeft;
        });
      });

      window.requestAnimationFrame(function () {
        isSyncing = false;
      });
    };

    const primaryTableId = tableIds[0];
    const primaryScrollables = scrollablesById[primaryTableId] || [];
    const primaryElement = primaryScrollables[0];

    if (primaryElement) {
      tableIds.slice(1).forEach(function (tableId) {
        (scrollablesById[tableId] || []).forEach(function (targetElement) {
          targetElement.scrollTop = primaryElement.scrollTop;
          targetElement.scrollLeft = primaryElement.scrollLeft;
        });
      });
    }

    const listeners = [];
    tableIds.forEach(function (tableId) {
      (scrollablesById[tableId] || []).forEach(function (element) {
        const handler = function () {
          syncScrollPosition(element, tableId);
        };
        element.addEventListener("scroll", handler, { passive: true });
        listeners.push({ element, handler });
      });
    });

    state.bindings[bindingKey] = {
      scrollablesById,
      listeners,
    };
  }

  function refreshBalanceTableEnhancements() {
    TABLE_GROUPS.forEach(function (tableIds) {
      bindScrollSyncGroup(tableIds);
    });
  }

  function scheduleRefresh() {
    if (state.rafId) {
      return;
    }

    state.rafId = window.requestAnimationFrame(function () {
      state.rafId = null;
      refreshBalanceTableEnhancements();
    });
  }

  function initialize() {
    scheduleRefresh();

    if (!state.observer) {
      state.observer = new MutationObserver(function () {
        scheduleRefresh();
      });

      if (document.body) {
        state.observer.observe(document.body, {
          childList: true,
          subtree: true,
        });
      }
    }

    window.addEventListener("resize", scheduleRefresh, { passive: true });
    window.setTimeout(scheduleRefresh, 150);
    window.setTimeout(scheduleRefresh, 400);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }

  window.addEventListener("load", scheduleRefresh, { passive: true });
})();
