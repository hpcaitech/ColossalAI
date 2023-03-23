当一个张量在上下游算子中被要求的sharding spec不同时，我们需要进行分布转换处理（Layout Conversion）。目前主流的方式有两种，打表转换和逐维度转换。打表转换就是将所有可能的情况枚举出来，然后在遇到需要转换的情况下，去表格中找到对应的转换方案。
为了解决这个问题，我们提出一个新奇的想法，使用启发式的搜索，来解决sharding spec的转换问题。
然而它有一个很大问题，就是随着设备块（Device Mesh）的维度增加，这个问题的规模极具膨胀，以至于无法通过这种枚举打表的方式来解决。逐维度转换是对于一个N-d tensor的sharding spec，X0X1...Xn-1，我们让i从0到n-1逐维度地进行转换，这样不管设备块和张量的维度多少，我们都只需要一次扫描，就可以得到一个可行的转换操作序列，然而它问题是这样的转换效率会很差。为了解决这个问题，我们提出一个新奇的想法，使用启发式算法，来解决sharding spec的转换问题。，这个算法可以描述为：
  1. 从source spec生成所有的one-step transform sharding specs
  2. 在one-step transform sharding specs中，根据相似度函数，挑选一个”区别最小“的sharding spec作为后续的source sharding spec，并将该sharding spec记录在transform path中，如果one-step transform sharding spec中，有与target sharding spec相同的sharding spec，则算法结束。
  3. 重复a，b直到算法结束

| Source/target sharding spec pairs |All gather | Shard | All to All | One step transform | Best sharding spec |Transform path|
| :-:         | :-:              | :-:                  | :-:                       | :-:                     | :-:                     |:-:                     |
| $S_{01}RR， RS_{01}R$  | $S_0RR$       | -           | $S_0RS_1, S_0S_1R$             | $S_0RR, S_0RS_1, S_0S_1R$             | $S_0RR$ | $S_0RR$
| $S_0RR, RS_{01}RR$  | $RRR$       | $S_0S_1R, S_0RS_1$           | $RS_0R, RRS_0$             | $RRR$, $S_0S_1R$, $S_0RS_1$, $RS_0R$, $RRS_0$             | $RS_0R$ | $S_0RR$ -> $RS_0R$
| $RS_0R, RS_{01}RR$  | $RRR$       | $RS_{01}R, S_1S_0R, RS_0S_1$           | $S_0RR, RRS_0$             | $RRR$, $RS_{01}R$, $S_1S_0R$, $RS_0S_1$, $S_0RR$, $RRS_0$             | $RS_{01}R$ | $S_0RR$ -> $RS_0R$ -> $RS_{01}R$
