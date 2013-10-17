uniform sampler2D realRGB;
uniform sampler2D realDepth;
uniform sampler2D virtualRGB;
uniform sampler2D virtualDepth;

void main (void)  
{

	vec4 realDepthColor = texture2D(realDepth, gl_TexCoord[0].st);
	vec4 virtualDepthColor = texture2D(virtualDepth, gl_TexCoord[0].st);
	vec4 realRGBColor = texture2D(realRGB, gl_TexCoord[0].st);
	vec4 virtualRGBColor = texture2D(virtualRGB, gl_TexCoord[0].st);
	
	/* //ARPolygonal
	if(virtualDepthColor.r == 1.0)
		gl_FragColor = realRGBColor;
	else if(virtualDepthColor.r <= realDepthColor.r)
		gl_FragColor = virtualRGBColor * 0.5 + realRGBColor * 0.5;	//blend effect
	else
		gl_FragColor = realRGBColor;
	*/

	//ARVolumetric
	if(virtualDepthColor.r == 1.0)
		gl_FragColor = realRGBColor;
	else if(virtualRGBColor.r == 0 && virtualRGBColor.g == 0 && virtualRGBColor.b == 0)
		gl_FragColor = realRGBColor;
	else if(virtualDepthColor.r <= realDepthColor.r)
		gl_FragColor = virtualRGBColor * 0.5 + realRGBColor * 0.5;
	else 
		gl_FragColor = realRGBColor;
}