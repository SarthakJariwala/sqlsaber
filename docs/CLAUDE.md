# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Development server**: `npm run dev` or `npm start`
- **Build documentation**: `npm run build`
- **Preview built site**: `npm run preview`

## Architecture

- **Documentation Site**: Built with Astro and Starlight for SQLsaber project documentation
- **Framework**: Astro 5.x with Starlight integration for documentation structure
- **Styling**: TailwindCSS 4.x with Starlight-specific Tailwind integration
- **Content**: MDX files in `src/content/docs/` for documentation pages
- **Assets**: Static images and SVGs in `src/assets/` and `public/`

## Project Context

This is the documentation website (`/docs`) for SQLsaber, an open-source agentic SQL assistant. The main SQLsaber Python project is in the parent directory with its own development workflow using uv/Python.

Key documentation areas:
- Landing page: `src/content/docs/index.mdx` (splash template with hero section)
- Guides: `src/content/docs/guides/` (tutorial content)
- Reference: `src/content/docs/reference/` (API and technical reference)

## Configuration Files

- `astro.config.mjs`: Main Astro configuration with Starlight integration
- `package.json`: Node.js dependencies and scripts
- `tsconfig.json`: TypeScript configuration for the Astro project
- Site URL configured for https://sqlsaber.com

## Content Structure

- Starlight sidebar automatically generates from `src/content/docs/` structure
- Use `.mdx` extension for components in markdown
- Use components available via `@astrojs/starlight/components`
- Asciinema integration for terminal recordings/demos
