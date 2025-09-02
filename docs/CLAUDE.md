## Commands

- **Start dev server**: `npm run dev`
- **Build for production**: `npm run build`
- **Preview production build**: `npm run preview`
- **Run Astro CLI**: `npm run astro`
- **Add new shadcn-svelte component**: `npx shadcn-svelte@latest add <ComponentName>`

## Architecture

This is the documentation website for SQLSaber, built with:

- **Astro**: Static site generator with hybrid SSG/SSR capabilities
- **Svelte**: Component framework for interactive UI elements
- **shadcn-svelte**: Pre-built component library with Tailwind CSS
- **TypeScript**: Full type safety across the codebase

### Project Structure

- `src/pages/`: Astro pages that define routes
- `src/lib/components/ui/`: shadcn-svelte UI components
- `src/lib/styles/`: Global CSS and Tailwind configuration
- `src/lib/utils.ts`: Utility functions for component styling
- `public/`: Static assets served directly

### Key Configuration Files

- `astro.config.mjs`: Astro configuration with Svelte integration
- `components.json`: shadcn-svelte component configuration
- `tsconfig.json`: TypeScript configuration with path aliases
- `svelte.config.js`: Svelte preprocessing configuration

## Development Guidelines

- Use `$lib/*` path aliases for importing from `src/lib/`
- Follow shadcn-svelte component patterns for UI consistency. If needed, feel free to add new shadcn-svelte components.
- Astro pages use `.astro` extension, components use `.svelte` unless you create Astro components
- When using Svelte, exclusively use Svelte 5 syntax only
- Follow Astro 5 best practices for splitting pages and components wherever appropriate
- CSS-in-JS via Tailwind classes, global styles in `src/lib/styles/app.css`
