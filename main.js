// Grab Tauri APIs
const { LogicalSize, LogicalPosition } = window.__TAURI__.window
const appWindow = window.__TAURI__.webviewWindow.getCurrentWebviewWindow()

// DOM refs
const app = document.getElementById("app")
const grid = document.getElementById("grid")
const chatContainer = document.getElementById("chat-container")
const chatInput = document.getElementById("chat-input")
const settingsPanel = document.getElementById("settings-panel")
const themeToggle = document.getElementById("theme-toggle")
const closeSettings = document.getElementById("close-settings")
const autoExpandToggle = document.getElementById("auto-expand-toggle")
const suggestionsToggle = document.getElementById("suggestions-toggle")
const saveConversationsToggle = document.getElementById("save-conversations-toggle")
const analyticsToggle = document.getElementById("analytics-toggle")
const backButton = document.getElementById("back-button")
const body = document.body

// State management
let originalGeometry = null
let isExpanded = false
let isExpanding = false
let hoverTimeout
let currentTheme = "light"
let settings = {
  autoExpand: true,
  suggestions: true,
  saveConversations: true,
  analytics: false,
}

// Capture original geometry with multiple attempts
async function captureOriginalGeometry() {
  if (originalGeometry) return originalGeometry

  try {
    console.log("Capturing original window geometry...")

    // Get current geometry
    const size = await appWindow.outerSize()
    const position = await appWindow.outerPosition()

    originalGeometry = {
      width: size.width,
      height: size.height,
      x: position.x,
      y: position.y,
    }

    console.log("Original geometry captured:", originalGeometry)

    // Store in localStorage as backup
    localStorage.setItem("artemis-original-geometry", JSON.stringify(originalGeometry))

    return originalGeometry
  } catch (error) {
    console.error("Error capturing geometry:", error)

    // Try to load from localStorage
    const saved = localStorage.getItem("artemis-original-geometry")
    if (saved) {
      originalGeometry = JSON.parse(saved)
      console.log("Loaded geometry from localStorage:", originalGeometry)
      return originalGeometry
    }

    // Fallback default
    originalGeometry = { width: 400, height: 300, x: 100, y: 100 }
    return originalGeometry
  }
}

// Theme management
function toggleTheme() {
  currentTheme = currentTheme === "light" ? "dark" : "light"
  updateAppMode()
  localStorage.setItem("artemis-theme", currentTheme)
  console.log(`Theme switched to: ${currentTheme}`)
}

// Initialize theme
function initTheme() {
  const savedTheme = localStorage.getItem("artemis-theme")
  if (savedTheme) {
    currentTheme = savedTheme
  }
  updateAppMode()
}

// Update app mode class
function updateAppMode() {
  const modeClass = isExpanded ? "mode-expanded" : "mode-compact"
  body.className = `theme-${currentTheme} ${modeClass}`

  // Update toggle states
  if (themeToggle) {
    themeToggle.classList.toggle("active", currentTheme === "dark")
  }
}

// Load settings
function loadSettings() {
  const savedSettings = localStorage.getItem("artemis-settings")
  if (savedSettings) {
    settings = { ...settings, ...JSON.parse(savedSettings) }
  }

  // Apply settings to toggles
  if (autoExpandToggle) {
    autoExpandToggle.classList.toggle("active", settings.autoExpand)
  }
  if (suggestionsToggle) {
    suggestionsToggle.classList.toggle("active", settings.suggestions)
  }
  if (saveConversationsToggle) {
    saveConversationsToggle.classList.toggle("active", settings.saveConversations)
  }
  if (analyticsToggle) {
    analyticsToggle.classList.toggle("active", settings.analytics)
  }
}

// Save settings
function saveSettings() {
  localStorage.setItem("artemis-settings", JSON.stringify(settings))
}

// Show chat on hover - ONLY when not expanded or expanding
app.addEventListener("mouseenter", () => {
  clearTimeout(hoverTimeout)
  if (!isExpanded && !isExpanding && settings.autoExpand) {
    chatContainer.classList.add("visible")
    grid.style.transform = "translateY(-8px)"
  }
})

// Hide chat when leaving - ONLY when not expanded or expanding
app.addEventListener("mouseleave", () => {
  if (!isExpanded && !isExpanding && settings.autoExpand) {
    hoverTimeout = setTimeout(() => {
      if (!isExpanded && !isExpanding) {
        chatContainer.classList.remove("visible")
        grid.style.transform = "translateY(0)"
      }
    }, 200)
  }
})

// Expand chat window - FLOATING TRANSPARENT WINDOW
async function expandChat(e) {
  e?.stopPropagation()
  if (isExpanded || isExpanding) return

  console.log("Expanding chat...")
  isExpanding = true

  // Capture original geometry if not already done
  await captureOriginalGeometry()

  // Clear any hover states immediately
  clearTimeout(hoverTimeout)
  chatContainer.classList.remove("visible")

  isExpanded = true
  updateAppMode()

  // Show elements for expanded mode
  chatContainer.classList.add("visible")
  grid.classList.add("hidden")

  // Calculate dimensions for floating chat
  const minWidth = 700
  const maxWidth = 1200
  const availWidth = screen.availWidth
  const newWidth = Math.min(maxWidth, Math.max(minWidth, availWidth * 0.8))
  const newHeight = 100

  // Center horizontally, position from top
  const x = Math.round((availWidth - newWidth) / 2)
  const y = Math.round(screen.availHeight * 0.2)

  console.log("Resizing to:", newWidth, "x", newHeight)
  console.log("Moving to:", x, ",", y)

  try {
    // Set window decorations to false for transparent effect
    await appWindow.setDecorations(false)

    // Resize and reposition window smoothly
    await Promise.all([
      appWindow.setSize(new LogicalSize(newWidth, newHeight)),
      appWindow.setPosition(new LogicalPosition(x, y)),
    ])

    console.log("Window resized and repositioned successfully")

    // Focus input after animations complete
    setTimeout(() => {
      chatInput.focus()
      isExpanding = false
    }, 300)
  } catch (error) {
    console.error("Error expanding chat:", error)
    isExpanding = false
  }
}

// Collapse chat window - FIXED with proper restoration
async function collapseChat() {
  if (!isExpanded) return

  console.log("Collapsing chat...")
  console.log("Restoring to original geometry:", originalGeometry)

  isExpanding = true
  isExpanded = false
  updateAppMode()

  // Update UI state
  chatContainer.classList.remove("visible")
  grid.classList.remove("hidden")

  try {
    // Ensure we have original geometry
    if (!originalGeometry) {
      console.warn("No original geometry available, using fallback")
      originalGeometry = { width: 400, height: 300, x: 100, y: 100 }
    }

    // Restore window decorations first
    await appWindow.setDecorations(true)

    // Small delay to ensure decorations are applied
    await new Promise((resolve) => setTimeout(resolve, 100))

    // Restore original size and position
    await appWindow.setSize(new LogicalSize(originalGeometry.width, originalGeometry.height))
    await new Promise((resolve) => setTimeout(resolve, 50))
    await appWindow.setPosition(new LogicalPosition(originalGeometry.x, originalGeometry.y))

    console.log("Window restored successfully")

    chatInput.blur()

    // Clear flag after restoration is complete
    setTimeout(() => {
      isExpanding = false
    }, 300)
  } catch (error) {
    console.error("Error collapsing chat:", error)
    isExpanding = false

    // Try fallback restoration
    try {
      await appWindow.setDecorations(true)
      await appWindow.setSize(new LogicalSize(400, 300))
      await appWindow.setPosition(new LogicalPosition(100, 100))
    } catch (fallbackError) {
      console.error("Fallback restoration also failed:", fallbackError)
    }
  }
}

// Settings panel management
function showSettings() {
  settingsPanel.classList.add("visible")
}

function hideSettings() {
  settingsPanel.classList.remove("visible")
}

// Event listeners for chat expansion
chatContainer.addEventListener("click", (e) => {
  e.stopPropagation()
  if (!isExpanded && !isExpanding) {
    expandChat(e)
  }
})

chatInput.addEventListener("focus", (e) => {
  if (!isExpanded && !isExpanding) {
    expandChat(e)
  }
})

// Back button event listener - IMPROVED
if (backButton) {
  backButton.addEventListener("click", (e) => {
    e.stopPropagation()
    console.log("Back button clicked")
    collapseChat()
  })
}

// Settings event listeners
if (themeToggle) {
  themeToggle.addEventListener("click", toggleTheme)
}

if (closeSettings) {
  closeSettings.addEventListener("click", hideSettings)
}

// Settings toggles
if (autoExpandToggle) {
  autoExpandToggle.addEventListener("click", () => {
    settings.autoExpand = !settings.autoExpand
    autoExpandToggle.classList.toggle("active", settings.autoExpand)
    saveSettings()
  })
}

if (suggestionsToggle) {
  suggestionsToggle.addEventListener("click", () => {
    settings.suggestions = !settings.suggestions
    suggestionsToggle.classList.toggle("active", settings.suggestions)
    saveSettings()
  })
}

if (saveConversationsToggle) {
  saveConversationsToggle.addEventListener("click", () => {
    settings.saveConversations = !settings.saveConversations
    saveConversationsToggle.classList.toggle("active", settings.saveConversations)
    saveSettings()
  })
}

if (analyticsToggle) {
  analyticsToggle.addEventListener("click", () => {
    settings.analytics = !settings.analytics
    analyticsToggle.classList.toggle("active", settings.analytics)
    saveSettings()
  })
}

// Click outside to collapse - IMPROVED
document.addEventListener("click", (e) => {
  // Handle settings panel
  if (settingsPanel.classList.contains("visible") && !settingsPanel.contains(e.target)) {
    hideSettings()
    return
  }

  // Handle chat expansion - but not if clicking back button or during expansion
  if (isExpanded && !isExpanding && !chatContainer.contains(e.target) && !backButton.contains(e.target)) {
    collapseChat()
  }
})

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (settingsPanel.classList.contains("visible")) {
      hideSettings()
    } else if (isExpanded && !isExpanding) {
      collapseChat()
    }
  }

  // Ctrl/Cmd + K to quick open chat
  if ((e.ctrlKey || e.metaKey) && e.key === "k") {
    e.preventDefault()
    if (!isExpanded && !isExpanding && !settingsPanel.classList.contains("visible")) {
      expandChat()
    }
  }

  // Ctrl/Cmd + , to open settings
  if ((e.ctrlKey || e.metaKey) && e.key === ",") {
    e.preventDefault()
    if (!settingsPanel.classList.contains("visible")) {
      showSettings()
    }
  }
})

// Enhanced shortcut interactions
document.querySelectorAll(".shortcut").forEach((shortcut) => {
  shortcut.addEventListener("click", (e) => {
    e.stopPropagation()
    const action = shortcut.dataset.action

    // Visual feedback animation
    shortcut.style.transform = "translateY(-2px) scale(0.95)"
    shortcut.style.transition = "transform 0.1s ease"

    setTimeout(() => {
      shortcut.style.transform = ""
      shortcut.style.transition = "all 0.35s cubic-bezier(0.4, 0, 0.2, 1)"
    }, 100)

    // Handle different actions
    switch (action) {
      case "chat":
        if (!isExpanded && !isExpanding) {
          expandChat()
        }
        break
      case "settings":
        showSettings()
        break
      case "focus":
        console.log("Focus mode clicked")
        break
      case "navigate":
        console.log("Navigate clicked")
        break
    }
  })

  // Enhanced hover effects for shortcuts (only in compact mode and not expanding)
  shortcut.addEventListener("mouseenter", () => {
    if (!isExpanded && !isExpanding) {
      shortcut.style.transform = "translateY(-2px) scale(1.02)"
    }
  })

  shortcut.addEventListener("mouseleave", () => {
    if (!isExpanded && !isExpanding) {
      shortcut.style.transform = ""
    }
  })
})

// Handle input submission
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault()
    const message = chatInput.value.trim()
    if (message) {
      console.log("Message sent:", message)

      // Simple message display
      const messageDisplay = document.createElement("div")
      messageDisplay.textContent = `✓ Sent: ${message}`
      messageDisplay.style.cssText = `
        position: absolute;
        top: -35px;
        left: 0;
        right: 0;
        text-align: center;
        font-size: 12px;
        font-weight: 500;
        color: ${currentTheme === "dark" ? "#10b981" : "#059669"};
        opacity: 0;
        transition: all 0.3s ease;
        pointer-events: none;
        background: ${currentTheme === "dark" ? "rgba(16, 185, 129, 0.1)" : "rgba(5, 150, 105, 0.1)"};
        padding: 6px 12px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
      `

      chatContainer.appendChild(messageDisplay)
      setTimeout(() => (messageDisplay.style.opacity = "1"), 10)
      setTimeout(() => {
        messageDisplay.style.opacity = "0"
        setTimeout(() => messageDisplay.remove(), 300)
      }, 2000)

      chatInput.value = ""
    }
  }
})

// Initialize the app
async function init() {
  console.log("Initializing app...")

  // Capture original geometry immediately
  await captureOriginalGeometry()

  loadSettings()
  initTheme()

  // Add startup animation
  setTimeout(() => {
    app.style.transition = "all 0.5s ease"
    app.style.opacity = "1"
    console.log("App initialized successfully")
  }, 100)
}

// Start the app
init()
