(function () {
  const TABLE_PAIRS = [
    {
      leftId: "balance-woodmac-table",
      rightId: "balance-comparison-delta-table",
    },
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

    existingBinding.leftListeners.forEach(function (listener) {
      listener.element.removeEventListener("scroll", listener.handler);
    });
    existingBinding.rightListeners.forEach(function (listener) {
      listener.element.removeEventListener("scroll", listener.handler);
    });
    delete state.bindings[bindingKey];
  }

  function bindScrollSync(leftId, rightId) {
    const leftScrollables = getScrollableElements(leftId);
    const rightScrollables = getScrollableElements(rightId);
    const bindingKey = `${leftId}::${rightId}`;
    const currentBinding = state.bindings[bindingKey];

    if (!leftScrollables.length || !rightScrollables.length) {
      cleanupBinding(bindingKey);
      return;
    }

    if (
      currentBinding &&
      currentBinding.leftScrollables.length === leftScrollables.length &&
      currentBinding.rightScrollables.length === rightScrollables.length &&
      currentBinding.leftScrollables.every(function (element, index) {
        return element === leftScrollables[index];
      }) &&
      currentBinding.rightScrollables.every(function (element, index) {
        return element === rightScrollables[index];
      })
    ) {
      return;
    }

    cleanupBinding(bindingKey);

    let isSyncing = false;

    const syncScrollPosition = function (sourceElement, targetElements) {
      if (isSyncing) {
        return;
      }

      isSyncing = true;
      targetElements.forEach(function (targetElement) {
        if (targetElement === sourceElement) {
          return;
        }

        targetElement.scrollTop = sourceElement.scrollTop;
        targetElement.scrollLeft = sourceElement.scrollLeft;
      });

      window.requestAnimationFrame(function () {
        isSyncing = false;
      });
    };

    rightScrollables.forEach(function (targetElement) {
      targetElement.scrollTop = leftScrollables[0].scrollTop;
      targetElement.scrollLeft = leftScrollables[0].scrollLeft;
    });

    const leftListeners = leftScrollables.map(function (element) {
      const handler = function () {
        syncScrollPosition(element, rightScrollables);
      };
      element.addEventListener("scroll", handler, { passive: true });
      return { element, handler };
    });

    const rightListeners = rightScrollables.map(function (element) {
      const handler = function () {
        syncScrollPosition(element, leftScrollables);
      };
      element.addEventListener("scroll", handler, { passive: true });
      return { element, handler };
    });

    state.bindings[bindingKey] = {
      leftScrollables,
      rightScrollables,
      leftListeners,
      rightListeners,
    };
  }

  function refreshBalanceTableEnhancements() {
    TABLE_PAIRS.forEach(function (pair) {
      bindScrollSync(pair.leftId, pair.rightId);
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
