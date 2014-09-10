#ifndef SH_H
#define SH_H

#include <math.h>
#define PI 3.14159265

namespace SH
{

	inline int computeBand(int index) 
	{
	
		if(index == 0)	return 0;
		else if(index > 0 && index < 4) return 1;
		else return 2;

	}

	inline int computeM(int index, int l) 
	{
		return index - l * (l + 1);
	}

	inline float factorial(int n) 
	{
		if (n <= 1)
		return(1);
		else
		return(n * factorial(n-1));
	}

	inline float DoubleFactorial(int n)
	{
		if (n <= 1)
		return(1);
		else
		return(n * DoubleFactorial(n-2));
	}

	static float Legendre(int l, int m, float x)
	{
		float result;
		if (l == m+1)
		result = x*(2*m + 1)*Legendre(m, m, x);
		else if (l == m)
		result = powf(-1, m)*DoubleFactorial(2 * m - 1)*powf((1 - x * x ), m/2);
		else
		result = (x*(2 * l - 1)*Legendre(l-1, m, x) - (l + m - 1)*Legendre(l-2, m, x))/(l-m);
		return(result);
	}

	static float K(int l, int m)
	{
		float num = (2*l+1) * factorial(l-abs(m));
		float denom = 4*PI * factorial(l+abs(m));
		float result = sqrt(num/denom);
		return(result);
	}

	static float SphericalHarmonics(int index, float theta, float phi) 
	{
		
		float result;
		int l = computeBand(index);
		int m = computeM(index, l);
		
		if (m > 0)
		result = sqrtf(2) * K(l, m) * cos(m*phi) * Legendre(l, m, cos(theta));
		else if (m < 0)
		result = sqrtf(2) * K(l, m) * sin(abs(m)*phi) * Legendre(l, abs(m), cos(theta));
		else
		result = K(l, m) * Legendre(l, 0, cos(theta));
		return(result);
		
		//result = sqrt(K(l, m)) * Legendre(l, m, cos(theta)) * cos(m * phi);
		//return result;
	}

};

#endif