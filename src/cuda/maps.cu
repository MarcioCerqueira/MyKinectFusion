/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 * 
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "device.hpp"

using namespace pcl::device;
using namespace pcl::gpu;

namespace pcl
{
  namespace device
  {
    __global__ void
    computeVmapKernel (const PtrStepSz<unsigned short> depth, PtrStep<float> vmap, float fx_inv, float fy_inv, float cx, float cy)
    {
      int u = threadIdx.x + blockIdx.x * blockDim.x;
      int v = threadIdx.y + blockIdx.y * blockDim.y;

      if (u < depth.cols && v < depth.rows)
      {
        int z = depth.ptr (v)[u];

        if (z != 0)
        {
          float vx = z * (u - cx) * fx_inv;
          float vy = z * (v - cy) * fy_inv;
          float vz = z;

          vmap.ptr (v                 )[u] = vx;
          vmap.ptr (v + depth.rows    )[u] = vy;
          vmap.ptr (v + depth.rows * 2)[u] = vz;
        }
        else
          vmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();

      }
    }

    __global__ void
    computeNmapKernel (int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
    {
      int u = threadIdx.x + blockIdx.x * blockDim.x;
      int v = threadIdx.y + blockIdx.y * blockDim.y;

      if (u >= cols || v >= rows)
        return;

      if (u == cols - 1 || v == rows - 1)
      {
        nmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();
        return;
      }

      float3 v00, v01, v10;
      v00.x = vmap.ptr (v  )[u];
      v01.x = vmap.ptr (v  )[u + 1];
      v10.x = vmap.ptr (v + 1)[u];

      if (!isnan (v00.x) && !isnan (v01.x) && !isnan (v10.x))
      {
        v00.y = vmap.ptr (v + rows)[u];
        v01.y = vmap.ptr (v + rows)[u + 1];
        v10.y = vmap.ptr (v + 1 + rows)[u];

        v00.z = vmap.ptr (v + 2 * rows)[u];
        v01.z = vmap.ptr (v + 2 * rows)[u + 1];
        v10.z = vmap.ptr (v + 1 + 2 * rows)[u];

        float3 r = normalized (cross (v01 - v00, v10 - v00));

        nmap.ptr (v       )[u] = r.x;
        nmap.ptr (v + rows)[u] = r.y;
        nmap.ptr (v + 2 * rows)[u] = r.z;
      }
      else
        nmap.ptr (v)[u] = numeric_limits<float>::quiet_NaN ();
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::createVMap (const Intr& intr, const DepthMap& depth, MapArr& vmap)
{
  vmap.create (depth.rows () * 3, depth.cols ());

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (depth.cols (), block.x);
  grid.y = divUp (depth.rows (), block.y);

  float fx = intr.fx, cx = intr.cx;
  float fy = intr.fy, cy = intr.cy;

  computeVmapKernel << < grid, block >> > (depth, vmap, 1.f / fx, 1.f / fy, cx, cy);
  cudaSafeCall (cudaGetLastError ());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::createNMap (const MapArr& vmap, MapArr& nmap)
{
  nmap.create (vmap.rows (), vmap.cols ());

  int rows = vmap.rows () / 3;
  int cols = vmap.cols ();

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  computeNmapKernel << < grid, block >> > (rows, cols, vmap, nmap);
  cudaSafeCall (cudaGetLastError ());
}

namespace pcl
{
  namespace device
  {
    __global__ void
    tranformMapsKernel (int rows, int cols, const PtrStep<float> vmap_src, const PtrStep<float> nmap_src,
                        const Mat33 Rmat, const float3 tvec, PtrStepSz<float> vmap_dst, PtrStep<float> nmap_dst, bool inverse)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      const float qnan = pcl::device::numeric_limits<float>::quiet_NaN ();

      if (x < cols && y < rows)
      {
        //vetexes
        float3 vsrc, vdst = make_float3 (qnan, qnan, qnan);
        vsrc.x = vmap_src.ptr (y)[x];

        if (!isnan (vsrc.x))
        {
          vsrc.y = vmap_src.ptr (y + rows)[x];
          vsrc.z = vmap_src.ptr (y + 2 * rows)[x];

		  if(!inverse)
			vdst = Rmat * vsrc + tvec;
		  else
			vdst = Rmat * (vsrc - tvec);

          vmap_dst.ptr (y + rows)[x] = vdst.y;
          vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;
        }

        vmap_dst.ptr (y)[x] = vdst.x;

        //normals
        float3 nsrc, ndst = make_float3 (qnan, qnan, qnan);
        nsrc.x = nmap_src.ptr (y)[x];

        if (!isnan (nsrc.x))
        {
          nsrc.y = nmap_src.ptr (y + rows)[x];
          nsrc.z = nmap_src.ptr (y + 2 * rows)[x];

          ndst = Rmat * nsrc;

          nmap_dst.ptr (y + rows)[x] = ndst.y;
          nmap_dst.ptr (y + 2 * rows)[x] = ndst.z;
        }

        nmap_dst.ptr (y)[x] = ndst.x;
      }
    }

	__global__ void
    tranformMapsKernel (int rows, int cols, const PtrStep<float> vmap_src, const PtrStep<float> nmap_src,
                        const Mat33 Rmat, const float3 tvec, PtrStepSz<float> vmap_dst, PtrStep<float> nmap_dst, 
						const float3 newOrigin, const float3 objectCenter)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      const float qnan = pcl::device::numeric_limits<float>::quiet_NaN ();

      if (x < cols && y < rows)
      {
        //vetexes
        float3 vsrc, vdst = make_float3 (qnan, qnan, qnan);
        vsrc.x = vmap_src.ptr (y)[x];

        if (!isnan (vsrc.x))
        {
			vsrc.y = vmap_src.ptr (y + rows)[x];
			vsrc.z = vmap_src.ptr (y + 2 * rows)[x];

			if(vsrc.z != 0) 
			{
			  //Convert to object's center
			  vsrc.x -= newOrigin.x;
			  vsrc.y -= newOrigin.y;
			  vsrc.z -= newOrigin.z;

			  vsrc.x -= objectCenter.x;
			  vsrc.y -= objectCenter.y;
			  vsrc.z -= objectCenter.z;

			  vdst = Rmat * vsrc + tvec;

			  /*
			  if(!inverseTransform) 
				vdst = Rmat * vsrc + tvec;
		      else
				vdst = Rmat * (vsrc - tvec);
			  */
			  
			  
			  vdst.x += objectCenter.x;
			  vdst.y += objectCenter.y;
			  vdst.z += objectCenter.z;

			  vdst.x += newOrigin.x;
			  vdst.y += newOrigin.y;
			  vdst.z += newOrigin.z;

		    
		     }

		  vmap_dst.ptr (y + rows)[x] = vdst.y;
		  vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;
		
		}

        vmap_dst.ptr (y)[x] = vdst.x;

        //normals
        float3 nsrc, ndst = make_float3 (qnan, qnan, qnan);
        nsrc.x = nmap_src.ptr (y)[x];

        if (!isnan (nsrc.x))
        {
          nsrc.y = nmap_src.ptr (y + rows)[x];
          nsrc.z = nmap_src.ptr (y + 2 * rows)[x];

          ndst = Rmat * nsrc;

          nmap_dst.ptr (y + rows)[x] = ndst.y;
          nmap_dst.ptr (y + 2 * rows)[x] = ndst.z;
        }

        nmap_dst.ptr (y)[x] = ndst.x;
      }
    }

  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::tranformMaps (const MapArr& vmap_src, const MapArr& nmap_src, 
                           const Mat33& Rmat, const float3& tvec, 
                           MapArr& vmap_dst, MapArr& nmap_dst, bool inverse)
{
  int cols = vmap_src.cols ();
  int rows = vmap_src.rows () / 3;

  vmap_dst.create (rows * 3, cols);
  nmap_dst.create (rows * 3, cols);

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  tranformMapsKernel << < grid, block >> > (rows, cols, vmap_src, nmap_src, Rmat, tvec, vmap_dst, nmap_dst, inverse);
  cudaSafeCall (cudaGetLastError ());

  cudaSafeCall (cudaDeviceSynchronize ());
}

void
pcl::device::tranformMaps (const MapArr& vmap_src, const MapArr& nmap_src, 
                           const Mat33& Rmat, const float3& tvec, 
                           MapArr& vmap_dst, MapArr& nmap_dst, const float3& newOrigin, const float3& objectCenter)
{
  int cols = vmap_src.cols ();
  int rows = vmap_src.rows () / 3;

  vmap_dst.create (rows * 3, cols);
  nmap_dst.create (rows * 3, cols);

  dim3 block (32, 8);
  dim3 grid (1, 1, 1);
  grid.x = divUp (cols, block.x);
  grid.y = divUp (rows, block.y);

  tranformMapsKernel << < grid, block >> > (rows, cols, vmap_src, nmap_src, Rmat, tvec, vmap_dst, nmap_dst, newOrigin, objectCenter);
  cudaSafeCall (cudaGetLastError ());

  cudaSafeCall (cudaDeviceSynchronize ());
}

namespace pcl
{
  namespace device
  {
    template<bool normalize>
    __global__ void
    resizeMapKernel (int drows, int dcols, int srows, const PtrStep<float> input, PtrStep<float> output)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= dcols || y >= drows)
        return;

      const float qnan = numeric_limits<float>::quiet_NaN ();

      int xs = x * 2;
      int ys = y * 2;

      float x00 = input.ptr (ys + 0)[xs + 0];
      float x01 = input.ptr (ys + 0)[xs + 1];
      float x10 = input.ptr (ys + 1)[xs + 0];
      float x11 = input.ptr (ys + 1)[xs + 1];

      if (isnan (x00) || isnan (x01) || isnan (x10) || isnan (x11))
      {
        output.ptr (y)[x] = qnan;
        return;
      }
      else
      {
        float3 n;

        n.x = (x00 + x01 + x10 + x11) / 4;

        float y00 = input.ptr (ys + srows + 0)[xs + 0];
        float y01 = input.ptr (ys + srows + 0)[xs + 1];
        float y10 = input.ptr (ys + srows + 1)[xs + 0];
        float y11 = input.ptr (ys + srows + 1)[xs + 1];

        n.y = (y00 + y01 + y10 + y11) / 4;

        float z00 = input.ptr (ys + 2 * srows + 0)[xs + 0];
        float z01 = input.ptr (ys + 2 * srows + 0)[xs + 1];
        float z10 = input.ptr (ys + 2 * srows + 1)[xs + 0];
        float z11 = input.ptr (ys + 2 * srows + 1)[xs + 1];

        n.z = (z00 + z01 + z10 + z11) / 4;

        if (normalize)
          n = normalized (n);

        output.ptr (y        )[x] = n.x;
        output.ptr (y + drows)[x] = n.y;
        output.ptr (y + 2 * drows)[x] = n.z;
      }
    }

    template<bool normalize>
    void
    resizeMap (const MapArr& input, MapArr& output)
    {
      int in_cols = input.cols ();
      int in_rows = input.rows () / 3;

      int out_cols = in_cols / 2;
      int out_rows = in_rows / 2;

      output.create (out_rows * 3, out_cols);

      dim3 block (32, 8);
      dim3 grid (divUp (out_cols, block.x), divUp (out_rows, block.y));
      resizeMapKernel<normalize><< < grid, block >> > (out_rows, out_cols, in_rows, input, output);
      cudaSafeCall ( cudaGetLastError () );
      cudaSafeCall (cudaDeviceSynchronize ());
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::resizeVMap (const MapArr& input, MapArr& output)
{
  resizeMap<false>(input, output);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::resizeNMap (const MapArr& input, MapArr& output)
{
  resizeMap<true>(input, output);
}

namespace pcl
{
  namespace device
  {

    template<typename T>
    __global__ void
    convertMapKernel (int rows, int cols, const PtrStep<float> map, PtrStep<T> output)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= cols || y >= rows)
        return;

      const float qnan = numeric_limits<float>::quiet_NaN ();

      T t;
      t.x = map.ptr (y)[x];
      if (!isnan (t.x))
      {
        t.y = map.ptr (y + rows)[x];
        t.z = map.ptr (y + 2 * rows)[x];
      }
      else
        t.y = t.z = qnan;

      output.ptr (y)[x] = t;
    }

	template<typename T>
    __global__ void
    convertErrorMapKernel (int rows, int cols, const PtrStep<float> map, PtrStep<T> output)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x >= cols || y >= rows)
        return;

      const float qnan = numeric_limits<float>::quiet_NaN ();

      T t;
      t.x = map.ptr (y)[x];
      if (!isnan (t.x))
      {
        t.y = map.ptr (y + rows)[x];
        t.z = map.ptr (y + 2 * rows)[x];
		t.w = map.ptr (y + 3 * rows)[x]; 
      }
      else
        t.y = t.z = t.w = qnan;

      output.ptr (y)[x] = t;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> void
pcl::device::convert (const MapArr& vmap, DeviceArray2D<T>& output)
{
  int cols = vmap.cols ();
  int rows = vmap.rows () / 3;

  output.create (rows, cols);

  dim3 block (32, 8);
  dim3 grid (divUp (cols, block.x), divUp (rows, block.y));

  convertMapKernel<T><< < grid, block >> > (rows, cols, vmap, output);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

template<typename T> void
pcl::device::convertXYZI (const MapArr& errorMap, DeviceArray2D<T>& output)
{
  int cols = errorMap.cols ();
  int rows = errorMap.rows () / 4;

  output.create (rows, cols);

  dim3 block (32, 8);
  dim3 grid (divUp (cols, block.x), divUp (rows, block.y));

  convertErrorMapKernel<T><< < grid, block >> > (rows, cols, errorMap, output);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

template void pcl::device::convert (const MapArr& vmap, DeviceArray2D<float4>& output);
template void pcl::device::convert<float8>(const MapArr& vmap, DeviceArray2D<float8>& output);
template void pcl::device::convertXYZI (const MapArr& vmap, DeviceArray2D<float4>& output);

