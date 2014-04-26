uniform sampler2D frontFrameBuffer;
uniform sampler2D backFrameBuffer;
uniform float stepSize;
uniform float clippingPlaneLeftX;
uniform float clippingPlaneRightX;
uniform float clippingPlaneUpY;
uniform float clippingPlaneDownY;
uniform float clippingPlaneFrontZ;
uniform float clippingPlaneBackZ;
uniform vec3 camera;
varying vec3 v;
uniform int windowWidth;
uniform int windowHeight;

bool checkClippingPlane(vec3 position) 
{
	if(position.x >= clippingPlaneLeftX && position.x < clippingPlaneRightX 
	&& position.y >= clippingPlaneDownY && position.y < clippingPlaneUpY
	&& position.z >= clippingPlaneFrontZ && position.z < clippingPlaneBackZ)
		return false;
	else
		return true;
}

void main (void)  
{

	vec4 rayStart = texture2D(frontFrameBuffer, vec2(gl_FragCoord.x/float(windowWidth), gl_FragCoord.y/float(windowHeight)));
	vec4 rayEnd = texture2D(backFrameBuffer, vec2(gl_FragCoord.x/float(windowWidth), gl_FragCoord.y/float(windowHeight)));
	
	if(rayStart == rayEnd)
		discard;
	
	//Initialize accumulated color and opacity
	vec4 dst = vec4(0, 0, 0, 0);
	//Determine volume entry position
	vec3 position = vec3(rayStart);
	vec3 direction = vec3(rayEnd) - vec3(rayStart);
	direction = normalize(direction);
	position = position + direction * stepSize;
	
	bool clip = checkClippingPlane(position);
	if(clip)
		dst = vec4(0.0, 0.0, 0.0, 1.0);
	else
		dst = vec4(1.0, 1.0, 1.0, 1.0);

	gl_FragColor = dst;

}