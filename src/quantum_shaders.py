"""
Shader programs for advanced quantum state visualization.
"""

# Vertex shader for surface plot with glow effect
SURFACE_VERTEX_SHADER = """
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;
layout(location = 2) in float probability;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

out vec2 vTexCoord;
out float vProbability;

void main() {
    vTexCoord = texCoord;
    vProbability = probability;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
"""

# Fragment shader for surface plot with glow effect
SURFACE_FRAGMENT_SHADER = """
#version 450
in vec2 vTexCoord;
in float vProbability;

uniform vec3 baseColor;
uniform float glowIntensity;

out vec4 fragColor;

void main() {
    // Calculate glow based on probability
    float glow = vProbability * glowIntensity;

    // Apply glow effect
    vec3 color = baseColor + glow * vec3(1.0, 0.8, 0.4);

    fragColor = vec4(color, 1.0);
}
"""

# Vertex shader for volumetric rendering
VOLUMETRIC_VERTEX_SHADER = """
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in float density;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

out vec3 vPosition;
out vec3 vNormal;
out float vDensity;

void main() {
    vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
    vNormal = normalMatrix * normal;
    vDensity = density;
    gl_Position = projectionMatrix * vec4(vPosition, 1.0);
}
"""

# Fragment shader for volumetric rendering
VOLUMETRIC_FRAGMENT_SHADER = """
#version 450
in vec3 vPosition;
in vec3 vNormal;
in float vDensity;

uniform vec3 lightPosition;
uniform vec3 cameraPosition;
uniform vec3 baseColor;
uniform float absorption;

out vec4 fragColor;

void main() {
    // Calculate lighting
    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightPosition - vPosition);
    vec3 V = normalize(cameraPosition - vPosition);
    vec3 H = normalize(L + V);

    // Calculate volumetric effects
    float density = clamp(vDensity, 0.0, 1.0);
    float transmittance = exp(-absorption * density);

    // Calculate final color
    float diffuse = max(dot(N, L), 0.0);
    float specular = pow(max(dot(N, H), 0.0), 32.0);

    vec3 color = baseColor * (diffuse + 0.2) + vec3(1.0) * specular;
    color = mix(color, baseColor, 1.0 - transmittance);

    fragColor = vec4(color, transmittance);
}
"""

# Vertex shader for ray tracing
RAYTRACING_VERTEX_SHADER = """
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;

out vec2 vTexCoord;
out vec3 vRayOrigin;
out vec3 vRayDirection;

uniform mat4 inverseProjection;
uniform mat4 inverseView;
uniform vec3 cameraPosition;

void main() {
    vTexCoord = texCoord;
    gl_Position = vec4(position, 1.0);

    // Calculate ray direction for ray tracing
    vec4 ray = inverseProjection * vec4(position.xy, 1.0, 1.0);
    ray = inverseView * vec4(ray.xyz, 0.0);
    vRayDirection = normalize(ray.xyz);
    vRayOrigin = cameraPosition;
}
"""

# Fragment shader for ray tracing
RAYTRACING_FRAGMENT_SHADER = """
#version 450
in vec2 vTexCoord;
in vec3 vRayOrigin;
in vec3 vRayDirection;

uniform sampler3D quantumState;
uniform float stepSize;
uniform int maxSteps;
uniform vec3 boundingBoxMin;
uniform vec3 boundingBoxMax;

out vec4 fragColor;

// Ray-box intersection test
bool intersectBox(vec3 origin, vec3 dir, out float tNear, out float tFar) {
    vec3 invDir = 1.0 / dir;
    vec3 tMin = (boundingBoxMin - origin) * invDir;
    vec3 tMax = (boundingBoxMax - origin) * invDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    tNear = max(max(t1.x, t1.y), t1.z);
    tFar = min(min(t2.x, t2.y), t2.z);
    return tNear <= tFar;
}

void main() {
    float tNear, tFar;
    if (!intersectBox(vRayOrigin, vRayDirection, tNear, tFar)) {
        fragColor = vec4(0.0);
        return;
    }

    // Ray marching through volume
    vec3 pos = vRayOrigin + tNear * vRayDirection;
    vec4 color = vec4(0.0);

    for (int i = 0; i < maxSteps && tNear < tFar; i++) {
        // Sample quantum state
        vec3 texCoord = (pos - boundingBoxMin) / (boundingBoxMax - boundingBoxMin);
        vec4 sample = texture(quantumState, texCoord);

        // Accumulate color and opacity
        color.rgb += (1.0 - color.a) * sample.rgb * sample.a;
        color.a += (1.0 - color.a) * sample.a;

        // Early ray termination
        if (color.a >= 0.99) break;

        // Step along ray
        pos += vRayDirection * stepSize;
        tNear += stepSize;
    }

    fragColor = color;
}
"""
