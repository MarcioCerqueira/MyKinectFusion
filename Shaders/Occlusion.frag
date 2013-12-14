uniform sampler2D realRGB;
uniform sampler2D realDepth;
uniform sampler2D virtualRGB;
uniform sampler2D virtualDepth;
uniform int ARPolygonal;

void main (void)  
{

	vec4 realDepthColor = texture2D(realDepth, vec2(gl_TexCoord[0].s, 1 - gl_TexCoord[0].t));
	vec4 virtualDepthColor = texture2D(virtualDepth, vec2(gl_TexCoord[0].s, 1 - gl_TexCoord[0].t));
	vec4 realRGBColor = texture2D(realRGB, gl_TexCoord[0].st);
	vec4 virtualRGBColor = texture2D(virtualRGB, vec2(gl_TexCoord[0].s, 1 - gl_TexCoord[0].t));
	
	realDepthColor.r = (1.0 - realDepthColor.r) * 100000.0;
	virtualDepthColor.r = (1.0 - virtualDepthColor.r) * 100000.0;
	
	if(virtualDepthColor.r == 0.0)
		gl_FragColor = realRGBColor;
	else if(virtualRGBColor.r == 0 && virtualRGBColor.g == 0 && virtualRGBColor.b == 0 && ARPolygonal == 0) //specific ARVolumetric
		gl_FragColor = realRGBColor;
	else if((virtualDepthColor.r + 0.025) > realDepthColor.r)
		gl_FragColor = virtualRGBColor * 0.6 + realRGBColor * 0.4;
	else
		gl_FragColor = realRGBColor;
	
}