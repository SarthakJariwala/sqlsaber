<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import { slide } from "svelte/transition";
  
  let mobileMenuOpen = $state(false);
</script>

<header
  class="border-b border-gray-100 bg-white/80 backdrop-blur-sm sticky top-0 z-50"
>
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    <div class="flex justify-between items-center h-16">
      <!-- Logo -->
      <div class="flex items-center">
        <a href="/" class="flex items-center space-x-2">
          <div
            class="w-8 h-8 bg-black rounded-md flex items-center justify-center"
          >
            <span class="text-white font-bold text-md">SQL</span>
          </div>
          <span class="font-semibold text-lg text-gray-900">saber</span>
        </a>
      </div>

      <!-- Navigation Links -->
      <nav class="hidden md:flex items-center space-x-8">
        <a
          href="https://github.com/SarthakJariwala/sqlsaber?tab=readme-ov-file#configuration"
          class="text-gray-600 hover:text-gray-900 transition-colors">docs</a
        >
      </nav>

      <!-- CTA Buttons -->
      <div class="hidden md:flex items-center space-x-4">
        <Button
          variant="ghost"
          class="text-gray-600 hover:text-gray-900"
          href="https://github.com/SarthakJariwala/sqlsaber"
        >
          GitHub
        </Button>
        <Button
          class="bg-black text-white"
          href="https://github.com/SarthakJariwala/sqlsaber"
        >
          Get Started</Button
        >
      </div>

      <!-- Mobile menu button -->
      <div class="md:hidden">
        <button
          class="text-gray-600 hover:text-gray-900 focus:outline-none focus:text-gray-900"
          aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
          onclick={() => mobileMenuOpen = !mobileMenuOpen}
        >
          <svg
            class="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d={mobileMenuOpen ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"}
            />
          </svg>
        </button>
      </div>
    </div>
  </div>

  <!-- Mobile menu panel -->
  {#if mobileMenuOpen}
    <div
      class="md:hidden bg-white/95 backdrop-blur-sm border-b border-gray-100"
      transition:slide={{ duration: 200 }}
    >
      <div class="px-4 py-3 space-y-3">
        <!-- Mobile Navigation Links -->
        <div class="space-y-2">
          <a
            href="https://github.com/SarthakJariwala/sqlsaber?tab=readme-ov-file#configuration"
            class="block px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-md transition-colors"
            onclick={() => mobileMenuOpen = false}
          >
            docs
          </a>
        </div>
        
        <!-- Mobile CTA Buttons -->
        <div class="pt-2 space-y-2 border-t border-gray-100">
          <Button
            variant="ghost"
            class="w-full justify-start text-gray-600 hover:text-gray-900"
            href="https://github.com/SarthakJariwala/sqlsaber"
          >
            GitHub
          </Button>
          <Button
            class="w-full bg-black text-white"
            href="https://github.com/SarthakJariwala/sqlsaber"
          >
            Get Started
          </Button>
        </div>
      </div>
    </div>
  {/if}
</header>
