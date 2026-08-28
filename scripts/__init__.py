# scripts 包标记：使 scripts 成为正规包，防止 site-packages 同名包遮蔽
# （notebook 的 `from scripts.xxx import ...` 依赖仓库根在 sys.path 首位）
