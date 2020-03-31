#version 150

uniform vec3 light_position_eye;
uniform float specular_exponent;
uniform float lighting_factor;
uniform vec4 fixed_color;

in vec3 position_eye;
in vec3 normal_eye;
in vec4 Kdi;
in vec4 Ksi;

out vec4 outColor;

vec3 Ls = 0.3*vec3(1.0);
vec3 Ld = vec3(1.0);

void main() {
    if(fixed_color != vec4(0.0)) {
		outColor = fixed_color;
		return;
	}

	float ni = 20.0; // number of intervals
	float t = Kdi.r;
	float r = max(0.0, 1.0 - 1.5*abs(t));
	if(t < 0) t += 2.0;
	float g = max(0.0, 1.0 - 1.5*abs(t-2./3.));
	float b = max(0.0, 1.0 - 1.5*abs(t-4./3.));
	float mul = (int(Kdi.g*ni) % 2) == 0 ? 1.0 : 0.5;
	vec3 Kdiq = mul * vec3(r, g, b);

	vec3 vector_to_light_eye = light_position_eye - position_eye;
	vec3 direction_to_light_eye = normalize(vector_to_light_eye);
	float dot_prod = dot(direction_to_light_eye, normal_eye);
	float clamped_dot_prod = max(dot_prod, 0.0);
	vec3 Id = Ld * Kdiq * clamped_dot_prod;    // Diffuse intensity

	vec3 reflection_eye = reflect(direction_to_light_eye, normal_eye);
	vec3 viewer_to_surface_eye = normalize(position_eye);
	float dot_prod_specular = dot(reflection_eye, viewer_to_surface_eye);
	dot_prod_specular = float(dot_prod >= 0) * max(dot_prod_specular, 0.0);
	float specular_factor = pow(dot_prod_specular, specular_exponent);
	vec3 Is = Ls * vec3(Ksi) * specular_factor;    // specular intensity

	outColor = vec4(lighting_factor * (Is + Id) + (1.0-lighting_factor) * Kdiq, Kdi.a);
}