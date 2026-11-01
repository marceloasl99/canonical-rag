---
name: spa-ux-accessibility
description: >-
  Audits single-page applications (SPA) for UX fluidity, accessibility (WCAG),
  theme transition integrity (dark/light mode), responsive design across devices,
  reading comfort, and non-blocking asynchronous UI feedback. Use when evaluating
  frontend interfaces, CSS/HTML quality, or interactive widgets.
---

# Single-Page Application (SPA) UX & Accessibility Skill

This skill provides a comprehensive review framework for reactive frontend interfaces, evaluating visual hierarchy, responsive layout, dark/light theme transitions, and accessibility.

## Core Review Areas

### 1. Contrast Ratios & Theme Consistency
- **WCAG AA / AAA Contrast**: Verify contrast ratios for body text (minimum 4.5:1), large text (minimum 3:1), and UI controls in both light mode and dark mode.
- **Palette Transitions**: Ensure transitions between themes (`transition: background-color ...`) do not cause layout shifts, flashes of unstyled content (FOUC), or unreadable inverted icons.
- **Colorblind Palettes**: Ensure status indicators (green for success, red for error, yellow for cache hit) are accompanied by descriptive icons or text labels.

### 2. Reactive State & Asynchronous Feedback
- **Zero Screen Freezing**: Confirm that long-running operations (query submission, indexing) display immediate visual feedback (such as stepped skeleton loaders) with clear cancellation or progress cues.
- **Error Toasts vs Native Alerts**: Ensure non-blocking toast stacks are used instead of disruptive browser `alert()` or `confirm()` modals.
- **Form Resets & Focus Management**: Verify that hitting Enter in input fields triggers submission gracefully, with focus returned to appropriate elements.

### 3. Reading Suite & Document Presentation
- **Full-Width Toggle**: Check that expanding reports ("Hide Sources") maintains sensible maximum line length (`max-w-prose` or comfortable reading measure) so lines don't stretch excessively on ultrawide monitors.
- **Typography Scale**: Ensure relative units (`rem`, `em`, Tailwind classes) allow font-size scaling without clipping text or breaking layout boxes.
- **Print Stylesheets**: Verify `@media print` rules hide sidebars, query controls, telemetry bars, and background colors to produce crisp, ink-saving printouts or PDFs.
