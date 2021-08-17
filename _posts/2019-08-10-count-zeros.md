---
title: Fast way to count zeros
author: Haytham ElFadeel
date: 2019-08-10 18:32:00 -0800

---


Population count is a procedure of counting the number of ones in a bit string. It is used in many applications such as Hamming distance, cardinality count in Bitarray, Binary Neural Networks and many other applications.

All major CPU vendors (Intel since 2008, AMD 2007, ARM in NEON) have such instruction in x86 it is called popcnt and was introduced in the SSE4.2 instruction set.

But before this instruction or if you don’t have access to this instruction the tradition way to count population is like this:

```csharp
public static int BitCount(ulong x)
{
    // put count of each 2 bits into those 2 bits
    x -= (x >> 1) & 0x5555555555555555UL;

    // put count of each 4 bits into those 4 bits
    x = (x & 0x3333333333333333UL) + ((x >> 2) & 0x3333333333333333UL);

    // put count of each 8 bits into those 8 bits
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FUL;

    // returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
    return (int) ((x * 0x0101010101010101UL) >> 56);
}

```

The good news is, in Java Integer.bitCount() and BitOperations.PopCount() / Popcnt.PopCount() in C# .NET Core are intrinsics. Which means they call the popcnt instruction directly.

But what if I told you there is a faster way to do it


## Instruction Latency and throughput
Every instruction has a latency or the number of CPU cycles it takes for the result to be available.

Throughput defines the maximum number of instructions of the same kind that can be executed per clock when the operand of each instruction is independent. The throughput is listed in the reciprocals of the throughputs. For example, a reciprocal throughput of 2 for FMUL means that a new FMUL instruction can start executing 2 clock cycles after a previous FMUL. A reciprocal throughput of 0.33 for ADD means that the execution units can handle 3 integer additions per clock cycle.

For modern Intel CPUs (e.g. Skylake) the latency of popcnt is 3 cycles, and the throughput 1 cycle.


## SIMD Population Count
While I was implementing Roaring Bitmap, I used the Popcnt.PopCount() intrinsic in .NET Core, but I wasn’t satisfied with the performance so I did some research and as usual Wojciech Muła had a SIMD solution in his great website http://0x80.pl/ which mean instead of operating on single machine word of 64bit I can operate on 128 or 256bit at a time.


## The Algorithm
While Wojciech Muła has two vectorized algorithms harley-seal and Muła. I will focus on the Muła algorithm.

The key ingredient is the AVX2 vector instruction vpshufb. The vpshufb instruction shuffles the input bytes into a new vector containing the same byte values in a potentially different order. It takes an input register v and a control mask m, treating both as vectors of 32 bytes. Starting from v0, v1, . . . , v31, it outputs a new vector (vm0 , vm1 , vm2 , vm3 , . . . , vm31 ) (assuming that 0 ≤ mi < 32 for i = 0, 1, . . . , 31). Thus, for example, if the mask m is 0, 1, 2, . . . , 31, then we have the identity function. If the mask m is 31, 30, . . . , 0, then the byte order is reversed. Bytes are allowed to be repeated in the output vector, thus the mask 0, 0, …, 0 would produce a vector containing only the first input byte, repeated 32 times. It is a fast instruction with a reciprocal throughput and latency of one cycle on current Intel processors, yet it effectively “looks up” 32 values at once.

In our case, we use a fixed input register made of the input bytes 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 corresponding to the population counts of all possible 4-bit integers 0, 1, 2, 3, . . . , 31. Given an array of 32 bytes, we can call vpshufb once, after selecting the least significant 4 bits of each byte (using a bitwise AND) to gather 32 population counts on 32 4-bit subwords. Next, we right shift by four bits each byte value, and call vpshufb again to gather 32 counts of the most significant 4 bits of each byte. We can sum the two results to obtain 32 population counts, each corresponding to one of the 32 initial byte values.

For small input size, the popcnt will be faster, but for input size of 256byte of more the Mula is faster, for my case where the size is 8KB it’s 2 time faster.

For C implantation go here: https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-avx2-lookup.cpp

Here is the C# .NET Core implementation for this algorithm using AVX2 intrinsic:

```csharp

private static readonly byte[] lookup_array = new byte[] { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };
private static readonly byte[] lookup_array = new byte[] { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };

// AVX2 implementation of Muła bit counting algorithm, ~2.7x faster than x64 PopCount
public unsafe static int BitCount_ptr_avx2(ulong* inputPtr)
{
    byte low_mask_val = 15;
	var low_mask = Avx2.BroadcastScalarToVector256(&low_mask_val);

    var lookup = Vector256<byte>.Zero;
    fixed (byte* lk = lookup_array)
        lookup = Avx2.LoadVector256(lk);

	var total = Vector256<ushort>.Zero;
	for (int i = 0; i < 1024; i += 4)
	{
    	var v = Avx2.LoadVector256(inputPtr + i).AsByte();

    	var lo = Avx2.And(v, low_mask);
    	var hi = Avx2.And(Avx2.ShiftRightLogical(v.AsUInt32(), 4).AsByte(), low_mask);

    	var popcnt1 = Avx2.Shuffle(lookup, lo);
    	var popcnt2 = Avx2.Shuffle(lookup, hi.AsByte());
    	var curTotal = Avx2.Add(popcnt1, popcnt2);
    	var curTotal2 = Avx2.SumAbsoluteDifferences(curTotal, Vector256<byte>.Zero);
    	total = Avx2.Add(curTotal2, total);
	}

	var buffer = new ulong[4];
	fixed (ulong* bufferPtr = buffer)
    	Avx2.Store(bufferPtr, total.AsUInt64());

    // final reduce
	ulong result = buffer[0] + buffer[1] + buffer[2] + buffer[3];
	return (int)result;
}



```
