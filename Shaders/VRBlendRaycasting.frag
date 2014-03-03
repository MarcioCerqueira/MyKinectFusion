uniform sampler3D volume;
uniform sampler3D minMaxOctree;
uniform sampler2D noise;
uniform sampler2D frontFrameBuffer;
uniform sampler2D backFrameBuffer;
uniform float stepSize;
uniform int clippingPlane;
uniform float clippingPlaneLeftX;
uniform float clippingPlaneRightX;
uniform float clippingPlaneUpY;
uniform float clippingPlaneDownY;
uniform float clippingPlaneFrontZ;
uniform float clippingPlaneBackZ;
uniform float earlyRayTerminationThreshold;
uniform vec3 camera;
uniform int stochasticJithering;
uniform int triCubicInterpolation;
uniform int MIP;
uniform int windowWidth;
uniform int windowHeight;

vec4 efficientTriCubicInterpolation(sampler3D texture, vec3 texCoord)
{

	vec3 texelSize;
	ivec3 voxelLength = textureSize(texture, 0);
	texelSize = 1.0 / vec3(voxelLength);

	vec3 coord_grid = texCoord * vec3(voxelLength) - 0.5;
	vec3 index = floor(coord_grid);
	vec3 fraction = coord_grid - index;

	vec3 one_frac = 1.0 - fraction;
	vec3 one_frac2 = one_frac * one_frac;
	vec3 fraction2 = fraction * fraction;

	vec3 w0 = (1.0/6.0) * one_frac2 * one_frac;
	vec3 w1 = (2.0/3.0) - 0.5 * fraction2 * (2.0 - fraction);
	vec3 w2 = (2.0/3.0) - 0.5 * one_frac2 * (2.0 - one_frac);
	vec3 w3 = (1.0/6.0) * fraction2 * fraction;

	vec3 g0 = w0 + w1;
	vec3 g1 = w2 + w3;
	vec3 h0 = texelSize * ((w1 / g0) - 0.5 + index);  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
	vec3 h1 = texelSize * ((w3 / g1) + 1.5 + index);

	vec4 tex000 = texture3D(texture, vec3(h0.x, h0.y, h0.z));
	vec4 tex100 = texture3D(texture, vec3(h1.x, h0.y, h0.z));
	tex000 = mix(tex000, tex100, g1.x);  //weigh along the x-direction
	vec4 tex010 = texture3D(texture, vec3(h0.x, h1.y, h0.z));
	vec4 tex110 = texture3D(texture, vec3(h1.x, h1.y, h0.z));
	tex010 = mix(tex010, tex110, g1.x);  //weigh along the x-direction
	tex000 = mix(tex000, tex010, g1.y);  //weigh along the y-direction
	vec4 tex001 = texture3D(texture, vec3(h0.x, h0.y, h1.z));
	vec4 tex101 = texture3D(texture, vec3(h1.x, h0.y, h1.z));
	tex001 = mix(tex001, tex101, g1.x);  //weigh along the x-direction
	vec4 tex011 = texture3D(texture, vec3(h0.x, h1.y, h1.z));
	vec4 tex111 = texture3D(texture, vec3(h1.x, h1.y, h1.z));
	tex011 = mix(tex011, tex111, g1.x);  //weigh along the x-direction
	tex001 = mix(tex001, tex011, g1.y);  //weigh along the y-direction
	return mix(tex000, tex001, g1.z);  //weigh along the z-direction	

}

bool checkClippingPlane(vec3 position) 
{
	if(position.x > clippingPlaneLeftX && position.x < clippingPlaneRightX 
	&& position.y > clippingPlaneDownY && position.y < clippingPlaneUpY
	&& position.z > clippingPlaneFrontZ && position.z < clippingPlaneBackZ)
		return false;
	else
		return true;
}

void main (void)  
{

	vec4 value = vec4(0, 0, 0, 0);
	float scalar;
	vec4 rayStart = texture2D(frontFrameBuffer, vec2(gl_FragCoord.x/float(windowWidth), gl_FragCoord.y/float(windowHeight)));
	vec4 rayEnd = texture2D(backFrameBuffer, vec2(gl_FragCoord.x/float(windowWidth), gl_FragCoord.y/float(windowHeight)));
	if(rayStart == rayEnd)
		discard;
	//Initialize accumulated color and opacity
	vec4 dst = vec4(0, 0, 0, 0);
	//Determine volume entry position
	vec3 position = vec3(rayStart);
	vec3 direction = vec3(rayEnd) - vec3(rayStart);
	float len = length(direction); // the length from front to back is calculated and used to terminate the ray
    direction = normalize(direction);
	if(stochasticJithering == 1)
		position = position + direction * texture2D(noise, gl_FragCoord.xy / 256.0).x/64.0;
	float dirLength = length(direction);
	//Loop for ray traversal
	float maxStepSize = 0.04; //2.f/50.f
	//float maxStepSize = 4.0 * stepSize;
	float accLength = 0.0;
	vec4 maxOpacity;
	bool clip = false;

	for(int i = 0; i < 200; i++) //Some large number
	{
		
		maxOpacity = texture3D(minMaxOctree, position);
		if(clippingPlane) clip = checkClippingPlane(position);
		if(maxOpacity.g > 0.0 && !clip) {
		
			//Data access to scalar value in 3D volume texture
			if(triCubicInterpolation == 1) {
				value = efficientTriCubicInterpolation(volume, position);
				vec3 s = vec3(-stepSize * 0.5, -stepSize * 0.5, -stepSize * 0.5);
				position = position + direction * s;
				value = efficientTriCubicInterpolation(volume, position);
				if(value.a > 0.1) s *= 0.5;
				else	s *= -0.5;
				position = position + direction * s;
				value = efficientTriCubicInterpolation(volume, position);
			} else {
				value = texture3D(volume, position);
				vec3 s = vec3(-stepSize * 0.5, -stepSize * 0.5, -stepSize * 0.5);
				position = position + direction * s;
				value = texture3D(volume, position);
				if(value.a > 0.1) s *= 0.5;
				else	s *= -0.5;
				position = position + direction * s;
				value = texture3D(volume, position);
			}

			if(MIP == 0) {
				if(value.a > 0.1)
					dst = (1.0 - dst.a) * value + dst;
			} else
				dst = max(dst, value);

			//Advance ray position along ray direction
			position = position + direction * stepSize;
			accLength += dirLength * stepSize;
		
		} else {
			
			position = position + direction * maxStepSize;
			accLength += dirLength * maxStepSize;
		
		}
		
		//Additional termination condition for early ray termination
		if(dst.a > earlyRayTerminationThreshold)
			break;
		
		//Ray termination: Test if outside volume...
		if(accLength > len)
			break;

	}

	gl_FragColor = dst;
	
}