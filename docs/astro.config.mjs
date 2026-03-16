// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightClientMermaid from '@pasqal-io/starlight-client-mermaid';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},
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
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/manucouto1/chronos-vector' },
			],
			customCss: [
				'./src/styles/custom.css',
			],
			head: [
				{
					tag: 'link',
					attrs: {
						rel: 'stylesheet',
						href: 'https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css',
						integrity: 'sha384-MkdEMsUo8Aj7/d0FPe2tUH4JXJrjHjyeO7OjSbgAVVqASjeFolb6ElTKjq73tL',
						crossorigin: 'anonymous',
					},
				},
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
						{ label: 'Interpretability', slug: 'architecture/interpretability' },
						{ label: 'Multi-Space & Multi-Scale', slug: 'architecture/multi-scale-alignment' },
						{ label: 'Temporal ML', slug: 'architecture/temporal-ml' },
						{ label: 'Data Virtualization', slug: 'architecture/data-virtualization' },
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
						{ label: 'Stochastic Processes', slug: 'research/stochastic-processes' },
						{ label: 'Path Signatures', slug: 'research/path-signatures' },
						{ label: 'Query Types', slug: 'research/query-types' },
						{ label: 'Tech Stack', slug: 'research/tech-stack' },
						{ label: 'Use Cases', slug: 'research/use-cases' },
						{ label: 'Competitive Landscape', slug: 'research/competitive-landscape' },
						{ label: 'Open Questions', slug: 'research/open-questions' },
					],
				},
				{
					label: 'Specifications',
					items: [
						{ label: 'Storage Layout', slug: 'specs/storage-layout' },
						{ label: 'API Contract', slug: 'specs/api-contract' },
						{ label: 'Benchmark Strategy', slug: 'specs/benchmark-plan' },
						{ label: 'Implementation Decisions', slug: 'specs/implementation-guide' },
						{ label: 'Git Workflow & Versioning', slug: 'specs/git-workflow' },
					],
				},
				{
					label: 'RFC',
					badge: { text: 'ADR', variant: 'note' },
					items: [
						{ label: 'RFC-001: Architecture Decisions', slug: 'rfc/rfc-001' },
						{ label: 'RFC-002: Correctness & Performance', slug: 'rfc/rfc-002' },
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
