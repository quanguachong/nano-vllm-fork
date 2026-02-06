from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    KV cache Block：表示一个固定大小的 KV cache 块（通常 256 tokens）。
    
    每个 Block 可以：
    - 被多个序列共享（通过 ref_count 引用计数）
    - 存储 token_ids 和 hash（用于 prefix cache 复用）
    """

    def __init__(self, block_id):
        self.block_id = block_id  # block 的唯一标识符
        # 【关键】引用计数：当多个序列共享同一个 block 时（prefix cache 复用），ref_count > 1
        # ref_count == 0 表示 block 空闲，可以重新分配
        self.ref_count = 0
        # 【关键】hash 值：用于 prefix cache 的快速查找
        # hash == -1 表示该 block 尚未计算 hash（通常是因为 block 未填满）
        self.hash = -1
        # 该 block 存储的 token_ids（用于 hash 计算和 prefix cache 匹配验证）
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """更新 block 的 hash 和 token_ids（通常在 block 填满时调用）"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置 block 状态（分配新 block 时调用）"""
        self.ref_count = 1  # 新分配的 block 引用计数为 1
        self.hash = -1  # 重置 hash
        self.token_ids = []  # 清空 token_ids


class BlockManager:
    """
    Block 管理器：负责 KV cache blocks 的分配、释放和 prefix cache 复用。
    
    核心功能：
    - PagedAttention 风格的块管理：固定大小的 blocks 池，按需分配
    - Prefix cache：通过 hash 匹配实现相同 prompt 前缀的 KV cache 复用
    - 引用计数：支持多个序列共享同一个 block（prefix cache 场景）
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size  # 每个 block 的大小（通常 256 tokens）
        # 【关键】blocks 池：预分配的所有 blocks（固定大小数组）
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 【关键】hash_to_block_id：prefix cache 的查找表
        # key 是 block 的 hash 值，value 是对应的 block_id
        # 用于快速查找是否存在相同的 block（token_ids 相同）
        self.hash_to_block_id: dict[int, int] = dict()
        # 【关键】free_block_ids：空闲 block 的 ID 队列（FIFO）
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used_block_ids：已使用的 block ID 集合（用于快速查找）
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        【关键】计算 block 的 hash 值（用于 prefix cache 匹配）
        
        Args:
            token_ids: 当前 block 的 token_ids
            prefix: 前一个 block 的 hash（链式 hash，确保前缀顺序敏感）
        
        Returns:
            block 的 hash 值（64 位整数）
        
        说明：
        - 如果 prefix != -1，会将前一个 block 的 hash 纳入计算（链式 hash）
        - 这样可以确保相同的前缀序列（多个连续 blocks）能被正确识别和复用
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 【关键】链式 hash：将前一个 block 的 hash 纳入计算
            # 这样可以区分相同 token_ids 但位置不同的情况
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        内部方法：分配一个空闲的 block
        
        Args:
            block_id: 要分配的 block ID
        
        Returns:
            分配后的 Block 对象
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0  # 确保 block 空闲
        block.reset()  # 重置 block 状态
        self.free_block_ids.remove(block_id)  # 从空闲队列移除
        self.used_block_ids.add(block_id)  # 加入已使用集合
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        内部方法：释放一个 block（将其标记为空闲）
        
        Args:
            block_id: 要释放的 block ID
        """
        assert self.blocks[block_id].ref_count == 0  # 确保引用计数为 0
        self.used_block_ids.remove(block_id)  # 从已使用集合移除
        self.free_block_ids.append(block_id)  # 加入空闲队列

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲 blocks 分配给序列"""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        【关键】为序列分配 KV cache blocks（prefill 阶段调用）
        
        核心逻辑：
        1. 遍历序列的每个 block，尝试通过 hash 匹配查找 prefix cache
        2. 如果命中 cache：复用已有 block，增加引用计数，累加 num_cached_tokens
        3. 如果未命中 cache：分配新 block，计算并存储 hash
        
        Args:
            seq: 需要分配 blocks 的序列
        """
        assert not seq.block_table  # 确保序列尚未分配 blocks
        h = -1  # 链式 hash 的前缀（前一个 block 的 hash）
        cache_miss = False  # 是否发生 cache miss（一旦 miss，后续 blocks 都 miss）
        
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)  # 获取第 i 个 block 的 token_ids
            
            # 【关键】计算 hash：只有当 block 完整（256 tokens）时才计算 hash
            # 不完整的 block（最后一个 block）不参与 prefix cache 匹配
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            
            # 尝试通过 hash 查找已存在的 block
            block_id = self.hash_to_block_id.get(h, -1)
            
            # 【关键】cache miss 判断：
            # 1. hash 未找到（block_id == -1）
            # 2. hash 找到但 token_ids 不匹配（hash 冲突或数据不一致）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            if cache_miss:
                # Cache miss：分配新 block
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 【关键】Cache hit：复用已有 block
                # 累加已缓存的 token 数（这些 token 不需要在 prefill 中重新计算）
                seq.num_cached_tokens += self.block_size
                
                if block_id in self.used_block_ids:
                    # Block 已被使用：增加引用计数（多个序列共享同一个 block）
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # Block 在 hash 表中但未使用：分配它（理论上不应该发生）
                    block = self._allocate_block(block_id)
            
            # 【关键】如果 hash 有效，更新 block 的 hash 和 token_ids，并写入查找表
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            # 将 block_id 加入序列的 block_table
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        【关键】释放序列占用的所有 KV cache blocks
        
        核心逻辑：
        1. 逆序遍历 block_table（从后往前），减少每个 block 的引用计数
        2. 当引用计数降为 0 时，释放 block（加入空闲队列）
        3. 重置序列的 num_cached_tokens 和 block_table
        
        Args:
            seq: 需要释放 blocks 的序列
        """
        # 【关键】逆序遍历：确保从最后一个 block 开始释放
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1  # 减少引用计数
            if block.ref_count == 0:
                # 引用计数为 0：没有其他序列使用该 block，可以释放
                self._deallocate_block(block_id)
        # 重置序列的缓存状态
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查序列是否可以追加新 token（decode 阶段调用）
        
        判断逻辑：
        - 当序列长度 % block_size == 1 时，需要分配新 block
        - 否则不需要新 block（当前 block 还有空间）
        
        Returns:
            True 如果可以追加（有足够的空闲 blocks 或不需要新 block）
        """
        # 【关键】判断是否需要新 block：
        # len(seq) % block_size == 1 表示当前 block 已满，下一个 token 需要新 block
        # 此时需要至少 1 个空闲 block；否则不需要新 block（返回 True）
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        【关键】为序列预分配下一个 block（decode 阶段调用）
        
        核心逻辑（根据序列当前长度决定）：
        1. len(seq) % block_size == 1：当前 block 刚满，需要分配新 block
        2. len(seq) % block_size == 0：当前 block 刚好填满，计算并存储 hash
        3. 其他情况：当前 block 未满，无需操作
        
        Args:
            seq: 需要追加 token 的序列
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]  # 最后一个 block
        
        if len(seq) % self.block_size == 1:
            # 【关键】情况 1：当前 block 刚满（第 257, 513, ... 个 token）
            # 需要分配新 block 用于存储下一个 token
            assert last_block.hash != -1  # 当前 block 应该有 hash
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)  # 将新 block 加入 block_table
            
        elif len(seq) % self.block_size == 0:
            # 【关键】情况 2：当前 block 刚好填满（第 256, 512, ... 个 token）
            # 计算并存储 hash，供后续 prefix cache 复用
            assert last_block.hash == -1  # 当前 block 应该还没有 hash
            token_ids = seq.block(seq.num_blocks-1)  # 获取最后一个 block 的 token_ids
            # 【关键】链式 hash：获取前一个 block 的 hash 作为前缀
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)  # 更新 block 的 hash 和 token_ids
            self.hash_to_block_id[h] = last_block.block_id  # 写入查找表
            
        else:
            # 情况 3：当前 block 未满，无需操作（下一个 token 仍在当前 block 内）
            assert last_block.hash == -1  # 未满的 block 不应该有 hash
