defmodule PdfOxide.MixProject do
  use Mix.Project

  def project do
    [
      app: :pdf_oxide,
      version: "0.3.68",
      elixir: "~> 1.15",
      compilers: [:elixir_make | Mix.compilers()],
      make_targets: ["all"],
      make_clean: ["clean"],
      deps: deps(),
      description: "Idiomatic Elixir bindings for pdf_oxide — fast PDF text/Markdown/HTML extraction.",
      package: [licenses: ["MIT"], links: %{"GitHub" => "https://github.com/yfedoseev/pdf_oxide"}]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp deps do
    [
      {:elixir_make, "~> 0.8", runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}
    ]
  end
end
