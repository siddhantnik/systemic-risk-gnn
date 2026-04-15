import { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Environment, Float, Box, Cylinder, Sphere, PresentationControls } from "@react-three/drei";

function SafeModel() {
  const groupRef = useRef();

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    // Subtle floating and tracking of pointer for the safe
    groupRef.current.rotation.y = Math.sin(t / 4) * 0.15 + (state.pointer.x * 0.25);
    groupRef.current.rotation.x = Math.sin(t / 3) * 0.08 - (state.pointer.y * 0.25);
  });

  return (
    <group ref={groupRef} dispose={null}>
      {/* Main Safe Body (Cube-like) */}
      <Box args={[3.4, 3.8, 3.4]} radius={0.05} castShadow receiveShadow>
        <meshStandardMaterial color="#18181A" roughness={0.65} metalness={0.7} />
      </Box>

      {/* Safe Door Outer Frame */}
      <Box args={[2.8, 3.2, 0.2]} position={[0, 0, 1.7]} castShadow receiveShadow>
        <meshStandardMaterial color="#222225" roughness={0.5} metalness={0.8} />
      </Box>

      {/* Safe Door Inner Bevel */}
      <Box args={[2.6, 3.0, 0.3]} position={[0, 0, 1.7]} castShadow receiveShadow>
        <meshStandardMaterial color="#1a1a1c" roughness={0.4} metalness={0.85} />
      </Box>

      {/* Handle Assembly */}
      <group position={[0, 0, 1.9]}>
        {/* Dial Base */}
        <Cylinder args={[0.7, 0.8, 0.25, 32]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <meshStandardMaterial color="#888890" roughness={0.15} metalness={0.95} />
        </Cylinder>
        
        {/* Center Dome */}
        <Sphere args={[0.35, 32, 32]} position={[0, 0, 0.15]} scale={[1, 1, 0.5]} castShadow>
          <meshStandardMaterial color="#ceced5" roughness={0.05} metalness={1.0} />
        </Sphere>

        {/* Rotary Spokes */}
        {[0, 1, 2, 3, 4].map((i) => {
          const angle = (i / 5) * Math.PI * 2;
          return (
            <group key={i} rotation={[0, 0, angle]}>
              <Cylinder args={[0.07, 0.07, 2.0, 16]} position={[0, 0.9, 0.1]} castShadow>
                <meshStandardMaterial color="#dfdfe5" roughness={0.1} metalness={1.0} />
              </Cylinder>
            </group>
          );
        })}
      </group>

      {/* Accents (Small glowing/metallic nodes on corners for tech feel) */}
      <Box args={[0.4, 0.4, 0.1]} position={[-1.1, 1.3, 1.85]} castShadow>
        <meshStandardMaterial color="#d3b47c" roughness={0.2} metalness={0.9} />
      </Box>
      <Box args={[0.4, 0.4, 0.1]} position={[1.1, -1.3, 1.85]} castShadow>
        <meshStandardMaterial color="#d3b47c" roughness={0.2} metalness={0.9} />
      </Box>
    </group>
  );
}

export default function VaultScene() {
  return (
    <Canvas
      camera={{ position: [0, 0, 8.5], fov: 45 }}
      shadows
      gl={{ alpha: true, antialias: true }}
      style={{ background: "transparent" }}
      onCreated={({ gl }) => {
        gl.setClearColor(0x000000, 0);      // transparent clear colour
        gl.clearColor(0, 0, 0, 0);
      }}
    >
      {/* Lighting for metallic glints */}
      <ambientLight intensity={0.5} />
      <spotLight 
        position={[6, 12, 10]} 
        intensity={2.8} 
        penumbra={1} 
        castShadow 
        angle={0.4} 
        color="#d3b47c" /* Champagne tint over vault */
      />
      <spotLight 
        position={[-6, 6, -6]} 
        intensity={1.5} 
        penumbra={1} 
        color="#ffffff" 
      />
      
      <PresentationControls
        global
        config={{ mass: 2, tension: 500 }}
        snap={{ mass: 4, tension: 1500 }}
        rotation={[0, 0.3, 0]}
        polar={[-Math.PI / 4, Math.PI / 4]}
        azimuth={[-Math.PI / 1.5, Math.PI / 2.5]}
      >
        <Float speed={2.5} rotationIntensity={0.15} floatIntensity={0.8}>
          <SafeModel />
        </Float>
      </PresentationControls>
      
      <Environment preset="city" background={false} />
    </Canvas>
  );
}
