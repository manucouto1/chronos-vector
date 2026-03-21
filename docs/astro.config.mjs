// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightClientMermaid from '@pasqal-io/starlight-client-mermaid';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
	site: 'https://manucouto1.github.io',
	base: '/chronos-vector',
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
						{ label: 'API Gateway', slug: 'architecture/api-gateway' },
						{ label: 'Concurrency Model', slug: 'architecture/concurrency' },
						{ label: 'Crate Structure', slug: 'architecture/crate-structure' },
						{ label: 'LLM Integration', slug: 'architecture/llm-integration', badge: { text: 'MCP', variant: 'success' } },
						{ label: 'Interpretability', slug: 'architecture/interpretability', badge: { text: 'Partial', variant: 'caution' } },
						{ label: 'Temporal ML', slug: 'architecture/temporal-ml', badge: { text: 'Partial', variant: 'caution' } },
						{ label: 'Multi-Space & Multi-Scale', slug: 'architecture/multi-scale-alignment', badge: { text: 'Partial', variant: 'caution' } },
						{ label: 'Observability', slug: 'architecture/observability', badge: { text: 'Planned', variant: 'note' } },
						{ label: 'Data Virtualization', slug: 'architecture/data-virtualization', badge: { text: 'Planned', variant: 'note' } },
						{ label: 'Deployment', slug: 'architecture/deployment', badge: { text: 'Planned', variant: 'note' } },
					],
				},
				{
					label: 'Applications',
					items: [
						{
							label: 'Mental Health & Clinical NLP',
							badge: { text: 'Validated', variant: 'success' },
							items: [
								{ label: 'Overview & Results', slug: 'applications/mental-health/overview' },
								{ label: 'Interactive Explorer (B1)', slug: 'tutorials/b1-explorer' },
								{ label: 'Clinical Anchoring (B2)', slug: 'tutorials/b2-clinical-anchoring' },
								{ label: 'Classification Benchmarks', slug: 'tutorials/b1-mental-health' },
							],
						},
						{
							label: 'Political Discourse',
							badge: { text: 'Validated', variant: 'success' },
							items: [
								{ label: 'Overview & Results', slug: 'applications/political-discourse/overview' },
							],
						},
						{
							label: 'AI Agent Memory',
							badge: { text: 'Active', variant: 'caution' },
							items: [
								{ label: 'Overview & Roadmap', slug: 'applications/agent-memory/overview' },
								{ label: 'Episodic Trace Memory', slug: 'research/episodic-trace-memory' },
								{ label: 'Experimental Report', slug: 'research/episodic-memory-experiments' },
							],
						},
						{
							label: 'Domain Showcase',
							badge: { text: 'Exploratory', variant: 'note' },
							items: [
								{ label: 'Overview', slug: 'applications/showcase/overview' },
								{ label: 'MAP-Elites Archive', slug: 'tutorials/map-elites' },
								{ label: 'MLOps Drift Detection', slug: 'tutorials/mlops-drift' },
								{ label: 'Market Regime Detection', slug: 'tutorials/finance-regimes' },
								{ label: 'Anomaly Detection (NAB)', slug: 'tutorials/nab-anomaly' },
							],
						},
					],
				},
				{
					label: 'Research',
					items: [
						{ label: 'Theoretical Foundations', slug: 'research/foundations' },
						{ label: 'Stochastic Processes', slug: 'research/stochastic-processes' },
						{ label: 'Path Signatures', slug: 'research/path-signatures' },
						{ label: 'Query Types', slug: 'research/query-types' },
						{ label: 'Competitive Landscape', slug: 'research/competitive-landscape' },
						{ label: 'Open Questions', slug: 'research/open-questions', badge: { text: 'Updated', variant: 'caution' } },
						{ label: 'White Paper', slug: 'research/white-paper' },
						{ label: 'Clinical Anchoring', slug: 'research/clinical-anchoring' },
						{ label: 'B1 Improvement Plan', slug: 'research/b1-improvement-plan' },
					],
				},
				{
					label: 'Tutorials',
					badge: { text: 'Guides', variant: 'tip' },
					items: [
						{ label: 'Quick Start', slug: 'tutorials/guides/quick-start' },
						{ label: 'Temporal Analytics', slug: 'tutorials/guides/temporal-analytics' },
						{ label: 'Anchor Projection & Centering', slug: 'tutorials/guides/anchor-projection' },
						{ label: 'Semantic Regions', slug: 'tutorials/guides/semantic-regions' },
						{ label: 'Episodic Memory for Agents', slug: 'tutorials/guides/episodic-memory' },
					],
				},
				{
					label: 'Specifications',
					items: [
						{ label: 'Python API Reference', slug: 'specs/python-api', badge: { text: '52 fns', variant: 'success' } },
						{ label: 'API & Examples', slug: 'examples/overview' },
						{ label: 'Storage Layout', slug: 'specs/storage-layout' },
						{ label: 'API Contract', slug: 'specs/api-contract' },
						{ label: 'Interactive API Reference', slug: 'specs/api-reference', badge: { text: 'OpenAPI', variant: 'success' } },
						{ label: 'Rust API Docs', slug: 'specs/rust-api', badge: { text: 'cargo doc', variant: 'note' } },
						{ label: 'Temporal Analytics Toolkit', slug: 'specs/temporal-analytics', badge: { text: '31 fns', variant: 'success' } },
						{ label: 'Performance Benchmarks', slug: 'specs/benchmarks', badge: { text: 'Results', variant: 'success' } },
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
						{ label: 'RFC-003: Neural ODE (TorchScript)', slug: 'rfc/rfc-003' },
						{ label: 'RFC-004: Semantic Regions', slug: 'rfc/rfc-004' },
						{ label: 'RFC-005: Query Capabilities', slug: 'rfc/rfc-005' },
						{ label: 'RFC-006: Anchor Projection', slug: 'rfc/rfc-006' },
						{ label: 'RFC-007: Temporal Primitives', slug: 'rfc/rfc-007', badge: { text: 'Done', variant: 'success' } },
						{ label: 'RFC-008: Index Architecture', slug: 'rfc/rfc-008', badge: { text: 'Done', variant: 'success' } },
						{ label: 'RFC-009: LLM Integration', slug: 'rfc/rfc-009', badge: { text: 'Done', variant: 'success' } },
						{ label: 'RFC-010: Temporal Graph', slug: 'rfc/rfc-010', badge: { text: 'Done', variant: 'success' } },
						{ label: 'RFC-011: Anchor Invariance', slug: 'rfc/rfc-011', badge: { text: 'Done', variant: 'success' } },
						{ label: 'RFC-012: Agent Memory & Performance', slug: 'rfc/rfc-012', badge: { text: 'New', variant: 'caution' } },
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
