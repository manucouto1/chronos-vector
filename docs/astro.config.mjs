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
						{ label: 'Interactive API Reference', slug: 'specs/api-reference', badge: { text: 'OpenAPI', variant: 'success' } },
						{ label: 'Rust API Docs', slug: 'specs/rust-api-docs' },
						{ label: 'Benchmark Strategy', slug: 'specs/benchmark-plan' },
						{ label: 'Performance Benchmarks', slug: 'specs/benchmarks', badge: { text: 'Results', variant: 'success' } },
						{ label: 'Temporal Analytics Toolkit', slug: 'specs/temporal-analytics', badge: { text: '17 fns', variant: 'success' } },
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
						{ label: 'RFC-003: Neural ODE (TorchScript)', slug: 'rfc/rfc-003' },
						{ label: 'RFC-004: Semantic Regions', slug: 'rfc/rfc-004' },
						{ label: 'RFC-005: Query Capabilities', slug: 'rfc/rfc-005' },
					],
				},
				{
					label: 'PRD',
					items: [
						{ label: 'Product Requirements', slug: 'prd/technical-prd' },
					],
				},
				{
					label: 'Examples',
					badge: { text: 'NEW', variant: 'tip' },
					items: [
						{ label: 'Overview', slug: 'examples/overview' },
						{ label: 'Mental Health Detection', slug: 'examples/mental-health', badge: { text: 'Live', variant: 'success' } },
						{ label: 'Quality-Diversity (MAP-Elites)', slug: 'examples/map-elites' },
						{ label: 'Molecular Dynamics', slug: 'examples/molecular-dynamics' },
						{ label: 'Drug Discovery', slug: 'examples/drug-discovery' },
						{ label: 'MLOps Drift Detection', slug: 'examples/mlops-drift' },
						{ label: 'Quantitative Finance', slug: 'examples/finance' },
					],
				},
				{
					label: 'Tutorials',
					badge: { text: 'Interactive', variant: 'tip' },
					items: [
						{ label: 'Overview', slug: 'tutorials/overview' },
						{ label: 'Mental Health Explorer', slug: 'tutorials/b1-explorer', badge: { text: 'Live', variant: 'success' } },
						{ label: 'MAP-Elites Archive', slug: 'tutorials/map-elites', badge: { text: 'Live', variant: 'success' } },
						{ label: 'MLOps Drift Detection', slug: 'tutorials/mlops-drift', badge: { text: 'Live', variant: 'success' } },
						{ label: 'B1: Classification Results', slug: 'tutorials/b1-mental-health' },
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
