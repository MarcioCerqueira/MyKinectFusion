uniform sampler2D realRGBTexture;
uniform sampler2D realDepthTexture;
uniform sampler2D virtualRGBTexture;
uniform sampler2D virtualDepthTexture;
uniform sampler2D curvatureMap;
uniform int ARPolygonal;
uniform int ARFromKinectFusionVolume;
uniform int ARFromVolumeRendering;
uniform int alphaBlending;
uniform int ghostViewBasedOnCurvatureMap;
uniform int ghostViewBasedOnDistanceFalloff;
uniform float curvatureWeight;
uniform float distanceFalloffWeight;
uniform vec2 focusPoint;
uniform float focusRadius;

vec4 computeFragmentColorARPolygonal(vec4 realDepth, vec4 virtualDepth, vec4 realRGB, vec4 virtualRGB, float alpha, float threshold) {
	
	if(virtualDepth.r == 0.0)
		return vec4(0, 0, 0, 0);
	else if((virtualDepth.r - threshold) < realDepth.r)
		return virtualRGB;
	else
		return vec4(0, 0, 0, 0);

}

vec4 computeFragmentColorARFromKinectFusionVolume(vec4 realDepth, vec4 virtualDepth, vec4 realRGB, vec4 virtualRGB, float alpha, float threshold) {

	if(virtualDepth.r == 0.0)
		return vec4(0, 0, 0, 0);
	else if(virtualRGB.r == 0.0 && virtualRGB.g == 0.0 && virtualRGB.b == 0.0)
		return vec4(0, 0, 0, 0);
	else if((virtualDepth.r - threshold) < realDepth.r)
		return virtualRGB;
	else
		return vec4(0, 0, 0, 0);

}

vec4 computeFragmentColorARFromVolumeRendering(vec4 realDepth, vec4 virtualDepth, vec4 realRGB, vec4 virtualRGB, float alpha, float threshold) {
	
	if(virtualDepth.r == 0.0)
		return vec4(0, 0, 0, 0);
	else if(virtualRGB.r == 0.0 && virtualRGB.g == 0.0 && virtualRGB.b == 0.0)
		return vec4(0, 0, 0, 0);
	else if((virtualDepth.r - threshold) < realDepth.r && distance(gl_FragCoord.xy, focusPoint) < focusRadius)
		return virtualRGB;
	else
		return vec4(0, 0, 0, 0);

}

void main (void)  
{

	vec4 realDepth = texture2D(realDepthTexture, vec2(gl_TexCoord[0].s, gl_TexCoord[0].t));
	vec4 virtualDepth = texture2D(virtualDepthTexture, vec2(gl_TexCoord[0].s, gl_TexCoord[0].t));
	
	vec4 realRGB = texture2D(realRGBTexture, gl_TexCoord[0].st);
	vec4 virtualRGB = texture2D(virtualRGBTexture, vec2(gl_TexCoord[0].s, gl_TexCoord[0].t));
	
	float threshold = 0.01;
	
	vec4 fragColor = virtualRGB;
	/*
	vec4 fragColor = vec4(0, 0, 0, 0);
	
	float alpha = 0;

	if(alphaBlending == 1)
		alpha = 0.2;
	else if(ghostViewBasedOnCurvatureMap == 1)
		alpha = texture2D(curvatureMap, gl_TexCoord[0].st).r * curvatureWeight;
	else if(ghostViewBasedOnDistanceFalloff == 1)
		alpha = max(alpha, pow(distance(gl_FragCoord.xy, focusPoint)/focusRadius, distanceFalloffWeight));
		
	alpha = clamp(alpha, 0.0, 1.0);

	if(ARPolygonal)
		fragColor = computeFragmentColorARPolygonal(realDepth, virtualDepth, realRGB, virtualRGB, alpha, threshold);
	else if(ARFromKinectFusionVolume)
		fragColor = computeFragmentColorARFromKinectFusionVolume(realDepth, virtualDepth, realRGB, virtualRGB, alpha, threshold);
	else {
		fragColor = computeFragmentColorARFromVolumeRendering(realDepth, virtualDepth, realRGB, virtualRGB, alpha, threshold);
	}
	*/
	gl_FragColor = fragColor;

}