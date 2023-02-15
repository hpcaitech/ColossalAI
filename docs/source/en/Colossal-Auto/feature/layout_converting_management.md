When a tensor is required to have different sharding specs in upstream and downstream operators, we need to perform layout conversion processing, which can also be called redistribution. There are currently two mainstream methods, enumeration conversion, and dimension-by-dimension conversion. enumeration conversion is to enumerate all possible situations, and then find the corresponding conversion scheme in the table when conversion is required. However, it has a big problem. That is, as the dimension of the device mesh increases, the scale of this problem is so inflated that it cannot be solved by enumerating tables. Dimension-by-dimension conversion is for a sharding spec of an N-D tensor, X0X1...Xn-1, sharding spec is converted from 0 to n-1 dimension by dimension, so that no matter how many dimensions the device mesh and tensor have, with only one-time Scanning, a feasible conversion operation sequence is generated, the problem is that the conversion efficiency will be very poor.

Therefore, we propose a novel algorithm, using heuristic search, to solve the conversion problem of sharding spec, which can be described as:
1. Generate all one-step transform sharding specs from source spec
2.  In the one-step transform sharding specs, according to the similarity function, select a sharding spec with the "least difference" as the subsequent source sharding spec, and record the sharding spec in the transform path. If a sharding spec of the one-step transforms is the same as the target sharding spec, the algorithm ends.
3. Repeat 1, 2 until the end of the algorithm


| Source/target sharding spec pairs |All gather | Shard | All to All | One step transform | Best sharding spec |Transform path|
| :-:         | :-:              | :-:                  | :-:                       | :-:                     | :-:                     |:-:                     |
| $S_{01}RR， RS_{01}R$  | $S_0RR$       | -           | $S_0RS_1, S_0S_1R$             | $S_0RR, S_0RS_1, S_0S_1R$             | $S_0RR$ | $S_0RR$
| $S_0RR, RS_{01}RR$  | $RRR$       | $S_0S_1R, S_0RS_1$           | $RS_0R, RRS_0$             | $RRR$, $S_0S_1R$, $S_0RS_1$, $RS_0R$, $RRS_0$             | $RS_0R$ | $S_0RR$ -> $RS_0R$
| $RS_0R, RS_{01}RR$  | $RRR$       | $RS_{01}R, S_1S_0R, RS_0S_1$           | $S_0RR, RRS_0$             | $RRR$, $RS_{01}R$, $S_1S_0R$, $RS_0S_1$, $S_0RR$, $RRS_0$             | $RS_{01}R$ | $S_0RR$ -> $RS_0R$ -> $RS_{01}R$
