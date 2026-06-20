%{
  configs: [
    %{
      name: "default",
      files: %{included: ["lib/", "test/"], excluded: []},
      strict: false,
      checks: %{enabled: []}
    }
  ]
}
