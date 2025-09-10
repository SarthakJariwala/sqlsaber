// @ts-check

import starlight from "@astrojs/starlight";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "astro/config";

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: "SQLsaber",
			customCss: ["./src/styles/global.css"],
			social: [
				{
					icon: "github",
					label: "GitHub",
					href: "https://github.com/SarthakJariwala/sqlsaber",
				},
			],
			sidebar: [
				{
					label: "Getting Started",
					items: [
						{ label: "Installation", slug: "installation" },
						{ label: "Quick Start", slug: "guides/getting-started" },
					],
				},
				{
					label: "Guides", 
					items: [
						{ label: "Database Setup", slug: "guides/database-setup" },
						{ label: "Authentication", slug: "guides/authentication" },
						{ label: "Models", slug: "guides/models" },
						{ label: "Running Queries", slug: "guides/queries" },
						{ label: "Conversation Threads", slug: "guides/threads" },
						{ label: "Memory Management", slug: "guides/memory" },
					],
				},
				{
					label: "Reference",
					items: [
						{ label: "Commands", slug: "reference/commands" },
					],
				},
				{
					label: "Project",
					items: [
						{ label: "Changelog", slug: "changelog" },
					],
				},
			],
		}),
	],
	vite: { plugins: [tailwindcss()] },

	site: "https://sqlsaber.com",
});
