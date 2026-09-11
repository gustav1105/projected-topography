use bevy::asset::{AssetLoader, LoadContext, RenderAssetUsages, io::Reader};
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use std::io::Cursor;
use thiserror::Error;
use tiff::decoder::{Decoder, DecodingResult, Limits};

#[derive(Default, TypePath)]
pub struct TifMeshLoader;

#[derive(Debug, Error)]
pub enum TifLoaderError {
    #[error("Failed to read asset buffer: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to decode TIF header: {0}")]
    Tiff(#[from] tiff::TiffError),
}

impl AssetLoader for TifMeshLoader {
    type Asset = Mesh;
    type Settings = ();
    type Error = TifLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;

        let cursor = Cursor::new(bytes);
        let mut decoder = Decoder::new(cursor)?.with_limits(Limits::unlimited());

        let (width, height) = decoder.dimensions()?;
        let result = decoder.read_image()?;

        let heights: Vec<f32> = match result {
            DecodingResult::F32(buf) => buf,
            DecodingResult::U16(buf) => buf.into_iter().map(|v| v as f32).collect(),
            DecodingResult::U8(buf) => buf.into_iter().map(|v| v as f32).collect(),
            _ => {
                return Err(TifLoaderError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Unsupported TIF bit depth",
                )));
            }
        };

        Ok(build_dem_mesh(
            width as usize,
            height as usize,
            &heights,
            3.5,
            4,
        ))
    }

    fn extensions(&self) -> &[&str] {
        &["tif", "tiff"]
    }
}

fn build_dem_mesh(
    width: usize,
    height: usize,
    heights: &[f32],
    vertical_scale: f32,
    step: usize,
) -> Mesh {
    let step = step.max(1);
    let grid_w = (width - 1) / step + 1;
    let grid_h = (height - 1) / step + 1;

    let mut positions = Vec::with_capacity(grid_w * grid_h);
    let mut uvs = Vec::with_capacity(grid_w * grid_h);
    let mut colors = Vec::with_capacity(grid_w * grid_h);
    let mut indices = Vec::with_capacity((grid_w - 1) * (grid_h - 1) * 6);

    let half_w = (width as f32) * 0.5;
    let half_h = (height as f32) * 0.5;

    let mut min_h = f32::MAX;
    let mut max_h = f32::MIN;
    for &h in heights.iter() {
        if h > -500.0 {
            min_h = min_h.min(h);
            max_h = max_h.max(h);
        }
    }
    if min_h == f32::MAX {
        min_h = 0.0;
    }

    let azimuth_rad = 315.0_f32.to_radians();
    let altitude_rad = 45.0_f32.to_radians();
    let lx = -altitude_rad.cos() * azimuth_rad.sin();
    let ly = altitude_rad.sin();
    let lz = -altitude_rad.cos() * azimuth_rad.cos();
    let light_dir = Vec3::new(lx, ly, lz).normalize();

    for gy in 0..grid_h {
        let y = (gy * step).min(height - 1);
        for gx in 0..grid_w {
            let x = (gx * step).min(width - 1);
            let idx = y * width + x;
            let raw_h = heights[idx];
            let h = if raw_h < -500.0 { min_h } else { raw_h };

            let rel_h = h - min_h;
            let y_pos = rel_h * vertical_scale;

            positions.push([(x as f32) - half_w, y_pos, (y as f32) - half_h]);
            uvs.push([
                x as f32 / (width - 1) as f32,
                y as f32 / (height - 1) as f32,
            ]);

            let x_left = x.saturating_sub(step);
            let x_right = (x + step).min(width - 1);
            let y_top = y.saturating_sub(step);
            let y_bottom = (y + step).min(height - 1);

            let hl = if heights[y * width + x_left] < -500.0 {
                min_h
            } else {
                heights[y * width + x_left]
            };
            let hr = if heights[y * width + x_right] < -500.0 {
                min_h
            } else {
                heights[y * width + x_right]
            };
            let ht = if heights[y_top * width + x] < -500.0 {
                min_h
            } else {
                heights[y_top * width + x]
            };
            let hb = if heights[y_bottom * width + x] < -500.0 {
                min_h
            } else {
                heights[y_bottom * width + x]
            };

            let dx = ((x_right - x_left) as f32).max(1.0);
            let dy = ((y_bottom - y_top) as f32).max(1.0);

            let dz_dx = (hr - hl) * vertical_scale / dx;
            let dz_dy = (hb - ht) * vertical_scale / dy;

            let normal = Vec3::new(-dz_dx, 1.0, -dz_dy).normalize();
            let shade = normal.dot(light_dir).clamp(0.0, 1.0);

            colors.push([shade, shade, shade, 1.0]);
        }
    }

    for gy in 0..(grid_h - 1) {
        for gx in 0..(grid_w - 1) {
            let top_left = (gy * grid_w + gx) as u32;
            let top_right = top_left + 1;
            let bottom_left = ((gy + 1) * grid_w + gx) as u32;
            let bottom_right = bottom_left + 1;

            indices.extend_from_slice(&[top_left, bottom_left, top_right]);
            indices.extend_from_slice(&[top_right, bottom_left, bottom_right]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();
    mesh
}

#[derive(Component)]
pub struct OrbitCamera {
    pub focus: Vec3,
    pub radius: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub rotate_sensitivity: f32,
    pub pan_sensitivity: f32,
    pub zoom_sensitivity: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: Vec3::ZERO,
            radius: 3000.0,
            pitch: 0.6, // ~35 degrees down angle
            yaw: 0.0,
            rotate_sensitivity: 0.005,
            pan_sensitivity: 1.5,
            zoom_sensitivity: 0.15,
        }
    }
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let terrain_handle: Handle<Mesh> = asset_server.load("terrain.tif");

    commands.spawn((
        Mesh3d(terrain_handle),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            unlit: true,
            ..default()
        })),
    ));

    // True Isometric Camera angles: Pitch = ~35.264 deg, Yaw = 45 deg
    let mut orbit = OrbitCamera::default();
    orbit.pitch = 35.264_f32.to_radians();
    orbit.yaw = 45.0_f32.to_radians();

    let initial_pos = Vec3::new(
        orbit.radius * orbit.yaw.sin() * orbit.pitch.cos(),
        orbit.radius * orbit.pitch.sin(),
        orbit.radius * orbit.yaw.cos() * orbit.pitch.cos(),
    );

    commands.spawn((
        Camera3d::default(),
        // 1. Switch to Orthographic projection
        Projection::Orthographic(OrthographicProjection {
            scale: 3.0, // Adjust initial zoom level (lower = zoomed in, higher = zoomed out)
            far: 50_000.0,
            near: -50_000.0,
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_translation(initial_pos).looking_at(orbit.focus, Vec3::Y),
        orbit,
    ));
}

fn camera_control_system(
    mouse_button: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    mut query: Query<(&mut OrbitCamera, &mut Transform, &mut Projection)>,
) {
    let Ok((mut orbit, mut transform, mut projection)) = query.single_mut() else {
        return;
    };

    let mouse_delta = mouse_motion.delta;
    let scroll_delta = mouse_scroll.delta.y;

    // 1. Orbit (Left Click Drag)
    if mouse_button.pressed(MouseButton::Left) {
        orbit.yaw -= mouse_delta.x * orbit.rotate_sensitivity;
        orbit.pitch += mouse_delta.y * orbit.rotate_sensitivity;
        orbit.pitch = orbit.pitch.clamp(0.05, 1.50);
    }

    // 2. Pan (Right Click Drag or Middle Click Drag)
    if mouse_button.pressed(MouseButton::Right) || mouse_button.pressed(MouseButton::Middle) {
        let right = transform.right();
        let up = transform.up();

        // Scale pan speed relative to orthographic zoom scale
        let current_scale = if let Projection::Orthographic(ref ortho) = *projection {
            ortho.scale
        } else {
            1.0
        };

        let pan_x = -mouse_delta.x * orbit.pan_sensitivity * current_scale;
        let pan_y = mouse_delta.y * orbit.pan_sensitivity * current_scale;

        orbit.focus += *right * pan_x + *up * pan_y;
    }

    // 3. Zoom via Orthographic Scale
    if scroll_delta != 0.0 {
        if let Projection::Orthographic(ref mut ortho) = *projection {
            let zoom_factor = 1.0 - scroll_delta * orbit.zoom_sensitivity;
            ortho.scale = (ortho.scale * zoom_factor).clamp(0.1, 20.0);
        }
    }

    // 4. Calculate final position & orientation
    let rot_quat = Quat::from_euler(EulerRot::YXZ, orbit.yaw, -orbit.pitch, 0.0);
    transform.translation = orbit.focus + rot_quat * Vec3::Z * orbit.radius;
    transform.look_at(orbit.focus, Vec3::Y);
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .register_asset_loader(TifMeshLoader)
        .add_systems(Startup, setup)
        .add_systems(Update, camera_control_system)
        .run();
}
