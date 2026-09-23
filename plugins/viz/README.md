# SQLsaber visualization plugin

The `sqlsaber-viz` plugin adds a `viz` tool. It builds a chart spec from query results and renders ASCII charts in the terminal with plotext.

```bash
uv tool install --with sqlsaber-viz sqlsaber
```

After a query, ask SQLsaber to plot the result. See [Plugins](https://sqlsaber.com/guides/plugins/).

SDK applications can configure the visualization model on the capability factory:

```python
from functools import partial

from sqlsaber import SQLSaberOptions
from sqlsaber_viz import capability

options = SQLSaberOptions(
    capabilities=[
        partial(
            capability,
            model_name="openai:gpt-5-mini",
            api_key="application-managed-key",
        )
    ]
)
```

Omit both arguments to inherit the session model and session API key.
