// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightClientMermaid from '@pasqal-io/starlight-client-mermaid';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'ChronosVector',
			description: 'High-Performance Temporal Vector Database in Rust',
			plugins: [starlightClientMermaid()],
			logo: {
				dark: './src/assets/logo-dark.svg',
				light: './src/assets/logo-light.svg',
				replacesTitle: false,
			},
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/manuelcoutopintos/chronos-vector' },
			],
			customCss: [
				'./src/styles/custom.css',
			],
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Introduction', slug: 'getting-started/introduction' },
						{ label: 'Vision & Motivation', slug: 'getting-started/vision' },
					],
				},
				{
					label: 'Architecture',
					items: [
						{ label: 'System Overview', slug: 'architecture/overview' },
						{ label: 'Subsystems', slug: 'architecture/subsystems' },
						{ label: 'Data Model', slug: 'architecture/data-model' },
						{ label: 'Ingestion Pipeline', slug: 'architecture/ingestion' },
						{ label: 'Temporal Index (ST-HNSW)', slug: 'architecture/temporal-index' },
						{ label: 'Tiered Storage', slug: 'architecture/tiered-storage' },
						{ label: 'Query Engine', slug: 'architecture/query-engine' },
						{ label: 'Analytics Engine', slug: 'architecture/analytics-engine' },
						{ label: 'API Gateway', slug: 'architecture/api-gateway' },
						{ label: 'Concurrency Model', slug: 'architecture/concurrency' },
						{ label: 'Crate Structure', slug: 'architecture/crate-structure' },
						{ label: 'Observability', slug: 'architecture/observability' },
						{ label: 'Deployment', slug: 'architecture/deployment' },
					],
				},
				{
					label: 'Research',
					items: [
						{ label: 'Theoretical Foundations', slug: 'research/foundations' },
						{ label: 'Query Types', slug: 'research/query-types' },
						{ label: 'Tech Stack', slug: 'research/tech-stack' },
						{ label: 'Competitive Landscape', slug: 'research/competitive-landscape' },
						{ label: 'Open Questions', slug: 'research/open-questions' },
					],
				},
				{
					label: 'Specifications',
					items: [
						{ label: 'Storage Layout', slug: 'specs/storage-layout' },
						{ label: 'API Contract', slug: 'specs/api-contract' },
					],
				},
				{
					label: 'RFC',
					badge: { text: 'ADR', variant: 'note' },
					items: [
						{ label: 'RFC-001: Architecture Decisions', slug: 'rfc/rfc-001' },
					],
				},
				{
					label: 'PRD',
					items: [
						{ label: 'Product Requirements', slug: 'prd/technical-prd' },
					],
				},
				{
					label: 'Roadmap',
					items: [
						{ label: 'Development Plan', slug: 'roadmap/iterative-plan' },
					],
				},
			],
		}),
	],
});
