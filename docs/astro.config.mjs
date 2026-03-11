// @ts-check

import sitemap from "@astrojs/sitemap";
import starlight from "@astrojs/starlight";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "astro/config";

// https://astro.build/config
export default defineConfig({
	integrations: [
		sitemap(),
		starlight({
			title: "SQLsaber",
			description:
				"Open-source AI SQL assistant. Ask questions in plain English and get SQL queries, results, and explanations. Supports PostgreSQL, MySQL, SQLite, and DuckDB.",
			customCss: ["./src/styles/global.css"],
			head: [
				{
					tag: "script",
					attrs: {
						defer: true,
						src: "https://umami.sarthakjariwala.com/script.js",
						"data-website-id": "72be4739-0d77-4d79-be5b-cf31a62bca87",
						"data-domains": "sqlsaber.com",
					},
				},
				{
					tag: "meta",
					attrs: {
						property: "og:image",
						content: "https://sqlsaber.com/og-image.png",
					},
				},
				{
					tag: "meta",
					attrs: {
						name: "twitter:card",
						content: "summary_large_image",
					},
				},
				{
					tag: "meta",
					attrs: {
						name: "twitter:image",
						content: "https://sqlsaber.com/og-image.png",
					},
				},
				{
					tag: "script",
					attrs: { type: "application/ld+json" },
					content: JSON.stringify({
						"@context": "https://schema.org",
						"@type": "SoftwareApplication",
						name: "SQLsaber",
						description:
							"Open-source AI SQL assistant. Ask questions in plain English and get SQL queries, results, and explanations.",
						applicationCategory: "DeveloperApplication",
						operatingSystem: "macOS, Linux, Windows",
						offers: { "@type": "Offer", price: "0" },
						url: "https://sqlsaber.com",
						downloadUrl: "https://pypi.org/project/sqlsaber/",
						softwareRequirements: "Python 3.12+",
						author: {
							"@type": "Person",
							name: "Sarthak Jariwala",
							url: "https://github.com/SarthakJariwala",
						},
					}),
				},
			],
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
						{ label: "Knowledge Base", slug: "guides/knowledge" },
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
					items: [{ label: "Changelog", slug: "changelog" }],
				},
			],
		}),
	],
	vite: { plugins: [tailwindcss()] },

	site: "https://sqlsaber.com",
});
