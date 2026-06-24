defmodule PdfOxide.MixProject do
  use Mix.Project

  def project do
    [
      app: :pdf_oxide,
      version: "0.3.69",
      elixir: "~> 1.15",
      compilers: [:elixir_make | Mix.compilers()],
      make_targets: ["all"],
      make_clean: ["clean"],
      deps: deps(),
      dialyzer: dialyzer(),
      description:
        "Idiomatic Elixir bindings for pdf_oxide — fast PDF text/Markdown/HTML extraction.",
      package: [
        licenses: ["MIT"],
        links: %{"GitHub" => "https://github.com/yfedoseev/pdf_oxide"},
        # Hex's default file set omits c_src/ and the Makefile — without them
        # elixir_make has nothing to build on the consumer's machine. List the
        # NIF sources explicitly so the published package can compile.
        files: ~w(lib c_src Makefile mix.exs README.md LICENSE CHANGELOG.md .formatter.exs)
      ]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp deps do
    [
      {:elixir_make, "~> 0.8", runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev], runtime: false}
    ]
  end

  # Keep the Dialyzer PLT under _build so CI can cache it with the build dir.
  defp dialyzer do
    [
      plt_local_path: "_build/#{Mix.env()}/dialyxir",
      plt_core_path: "_build/#{Mix.env()}/dialyxir"
    ]
  end
end
