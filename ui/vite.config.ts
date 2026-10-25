import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    cssCodeSplit: false,
    lib: { entry: 'src/index.tsx', formats: ['es'], fileName: () => 'twm.js' },
    rollupOptions: { output: { inlineDynamicImports: true, assetFileNames: 'twm.css' } },
    minify: 'esbuild',
  },
});
