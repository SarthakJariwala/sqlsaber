<script lang="ts">
  interface Props {
    title: string;
    description: string;
    features?: string[];
    reversed?: boolean;
    codeExample?: string | null;
    className?: string;
  }

  let {
    title,
    description,
    features = [],
    reversed = false,
    codeExample = null,
    className = "",
  }: Props = $props();
</script>

<section class="py-20 px-4 sm:px-6 lg:px-8 {className}">
  <div class="max-w-6xl mx-auto">
    <div class="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
      <!-- Content -->
      <div class="space-y-8 {reversed ? 'lg:order-2' : ''}">
        <div>
          <h2
            class="text-4xl lg:text-5xl font-bold text-gray-900 mb-6 leading-tight"
          >
            {title}
          </h2>
          <div class="text-lg text-gray-600 leading-relaxed space-y-4">
            {@html description}
          </div>
        </div>

        {#if features.length > 0}
          <ul class="space-y-4">
            {#each features as feature}
              <li class="flex items-start space-x-3">
                <div
                  class="flex-shrink-0 w-6 h-6 bg-black rounded-full flex items-center justify-center mt-0.5"
                >
                  <svg
                    class="w-3 h-3 text-white"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fill-rule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clip-rule="evenodd"
                    />
                  </svg>
                </div>
                <span class="text-gray-700 font-medium">{feature}</span>
              </li>
            {/each}
          </ul>
        {/if}
      </div>

      <!-- Visual/Code Example -->
      <div class={reversed ? "lg:order-1" : ""}>
        {#if codeExample}
          <div class="bg-white border rounded p-4 text-left">
            <span class="text-gray-400">$</span>
            <span>{@html codeExample}</span>
          </div>
        {:else}
          <script
            src="https://asciinema.org/a/YITbKtH4mPn8LLTr6QEUh7e8f.js"
            id="asciicast-YITbKtH4mPn8LLTr6QEUh7e8f"
            async="true"
          ></script>
        {/if}
      </div>
    </div>
  </div>
</section>
