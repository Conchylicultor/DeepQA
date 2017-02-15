You can create your own dataset using a simple custom format where one line correspond to one line of dialogue. Use `===` to separate conversations between 2 people. Example of conversation file:


```
from A to B
from B to A
from A to B
from B to A
from A to B
===
from C to D
from D to C
from C to D
===
from E to F
from F to E
from E to F
from F to E
```

To use your conversation file `<name>.txt`, copy it in this repository and launch the program with the option `--corpus lightweight --datasetTag <name>`.
